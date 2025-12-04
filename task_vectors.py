"""
Task Vector classes following TVQ reference implementation.

Provides utilities for working with task vectors (parameter deltas between
fine-tuned and pretrained models) and quantized representations.

=== TUTORIAL: Understanding Task Vectors ===

A task vector represents what a model "learned" during fine-tuning on a specific task.
It's simply the difference between the fine-tuned model's weights and the base model's weights:

    task_vector = θ_finetuned - θ_pretrained

=== WHY TASK VECTORS ARE USEFUL ===

1. **Model Merging**: Combine multiple task-specific models into one by averaging task vectors
2. **Transfer Learning**: Apply one task's learning to another model
3. **Storage Efficiency**: Store task vectors instead of full models (only store deltas)
4. **Arithmetic Operations**: Add, subtract, and scale task vectors to control model behavior

=== KEY CLASSES ===

1. **TaskVector**: Basic task vector (full precision deltas)
   - Computed from pretrained and fine-tuned checkpoints
   - Supports arithmetic operations (+, -, *)
   - Can be applied to a base model to reconstruct the fine-tuned model

2. **QuantizedTaskVector**: Task vector with quantized deltas
   - Stores deltas as low-bit integers for compression
   - Can be dequantized and applied to a model

3. **QuantizedFinetunedModel**: Store fine-tuned model as quantized weights
   - Quantizes the entire model, not just deltas
   - Can compute task vectors on demand

4. **QuantizedBaseAndTaskVector**: Store both base and delta quantized
   - Maximum compression: both parts quantized
   - Useful when base model is also large

=== EXAMPLE WORKFLOW ===

    >>> from task_vectors import TaskVector
    >>> 
    >>> # Create task vectors from checkpoints
    >>> tv_cars = TaskVector("base.pt", "finetuned_cars.pt", task_name="Cars")
    >>> tv_flowers = TaskVector("base.pt", "finetuned_flowers.pt", task_name="Flowers")
    >>> 
    >>> # Average task vectors for multi-task model
    >>> tv_merged = tv_cars * 0.5 + tv_flowers * 0.5
    >>> 
    >>> # Apply to base model to get merged model
    >>> merged_state_dict = tv_merged.apply_to("base.pt")
"""

import torch
from typing import Dict, Optional, Union
from pathlib import Path
import quantization_utils


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    Extract state dict from a checkpoint, handling various formats.
    
    This helper function handles:
    1. Direct state dicts (dict with tensor values)
    2. Nested dicts with "state_dict", "model", or "model_state_dict" keys
    3. Full model objects (torch.nn.Module) by calling .state_dict()
    
    Args:
        checkpoint: Loaded checkpoint (could be dict, state dict, or model object)
        
    Returns:
        State dictionary mapping parameter name to tensor
    """
    # Handle torch.nn.Module objects (full model saved via torch.save(model, path))
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    
    # Handle dictionary formats
    if isinstance(checkpoint, dict):
        # Check for nested state dict under various keys
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        else:
            # Assume it's already a state dict
            return checkpoint
    
    # If it's neither a Module nor a dict, it might be a state dict directly
    # (though this is unusual, return as-is and let caller handle)
    return checkpoint


class TaskVector:
    """
    Represents a task vector (delta = finetuned - pretrained).
    
    A task vector captures what a model learned during fine-tuning by storing
    the difference between the fine-tuned and pretrained weights for each parameter.
    
    === HOW IT WORKS ===
    
    For each parameter in the model:
        delta[param] = finetuned_model[param] - pretrained_model[param]
    
    === SUPPORTED OPERATIONS ===
    
    - **Addition**: tv1 + tv2 → Combined task vector (model learns both tasks)
    - **Subtraction**: tv1 - tv2 → Difference in what was learned
    - **Scaling**: tv * 0.5 → Scale the magnitude of changes (regularization)
    - **Apply**: tv.apply_to(base_model) → Reconstruct fine-tuned model
    
    === TYPICAL USAGE ===
    
        # Single task
        >>> tv = TaskVector("base.pt", "finetuned.pt")
        >>> reconstructed = tv.apply_to("base.pt")  # Gets back finetuned model
        
        # Task arithmetic (merge two tasks)
        >>> tv1 = TaskVector("base.pt", "task1_finetuned.pt")
        >>> tv2 = TaskVector("base.pt", "task2_finetuned.pt")
        >>> tv_merged = tv1 * 0.5 + tv2 * 0.5  # Average
        >>> merged_model = tv_merged.apply_to("base.pt")
        
        # Task negation (remove a capability)
        >>> tv_general = TaskVector("base.pt", "general_finetuned.pt")
        >>> tv_toxic = TaskVector("base.pt", "toxic_finetuned.pt")
        >>> tv_safe = tv_general - tv_toxic * 0.5  # Reduce toxic capability
    
    Attributes:
        vector: Dictionary mapping parameter name -> delta tensor
        task_name: Optional task identifier (for debugging/logging)
    """
    
    def __init__(
        self,
        pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]],
        finetuned_checkpoint: Union[str, Dict[str, torch.Tensor]],
        task_name: Optional[str] = None,
        skip_int64: bool = True,
        skip_uint8: bool = True,
        verbose: bool = True
    ):
        """
        Initialize task vector from checkpoints.
        
        Loads both checkpoints and computes the delta (difference) for each
        shared parameter. Parameters that exist in only one checkpoint are ignored.
        
        Args:
            pretrained_checkpoint: Path to checkpoint file (str) or state dict (dict)
            finetuned_checkpoint: Path to checkpoint file (str) or state dict (dict)
            task_name: Optional name for this task (used in logging and combined names)
            skip_int64: Skip int64 parameters (typically buffers, not learnable)
            skip_uint8: Skip uint8 parameters (typically quantized or index tensors)
            verbose: If True, print educational tutorial-style progress messages with validation
            
        Example:
            # From file paths
            >>> tv = TaskVector("./base_model.pt", "./finetuned_on_cars.pt", task_name="Cars")
            
            # From already-loaded state dicts
            >>> tv = TaskVector(base_state_dict, finetuned_state_dict)
        """
        self.task_name = task_name
        self.verbose = verbose
        
        if self.verbose:
            task_str = f" for task '{task_name}'" if task_name else ""
            print(f"\n{'='*70}")
            print(f"📚 TUTORIAL: Creating a Task Vector{task_str}")
            print(f"{'='*70}")
            print(f"""
🎯 WHAT IS A TASK VECTOR?
   A task vector captures what a model "learned" during fine-tuning.
   It's simply the difference between fine-tuned and original weights:
   
   task_vector = fine_tuned_model - pretrained_model
   
   Think of it like this: if the pretrained model is a "blank slate",
   the task vector represents the "knowledge" added during training.
""")
        
        # Step 1: Load pretrained checkpoint
        if isinstance(pretrained_checkpoint, str):
            if self.verbose:
                print(f"📂 STEP 1: Loading the Pretrained (Base) Model")
                print(f"   └─ This is the original model before any task-specific training")
                print(f"   └─ File: {pretrained_checkpoint}")
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
            if self.verbose:
                # Validation: Check if loaded successfully
                if pretrained_state is not None:
                    print(f"   ✅ VALIDATION PASSED: File loaded successfully")
                else:
                    print(f"   ❌ VALIDATION FAILED: File loaded but returned None")
        else:
            if self.verbose:
                print(f"📂 STEP 1: Using Provided Pretrained State Dict/Model")
                print(f"   └─ The pretrained model/weights were passed directly")
            pretrained_state = pretrained_checkpoint
            if self.verbose:
                if pretrained_state is not None:
                    # Check if it's a dict or model object
                    if isinstance(pretrained_state, dict):
                        if len(pretrained_state) > 0:
                            print(f"   ✅ VALIDATION PASSED: State dict is valid and non-empty")
                        else:
                            print(f"   ⚠️ WARNING: State dict appears empty")
                    elif hasattr(pretrained_state, 'state_dict'):
                        print(f"   ✅ VALIDATION PASSED: Model object detected (will extract state dict)")
                    else:
                        print(f"   ✅ VALIDATION PASSED: Data received (will attempt extraction)")
                else:
                    print(f"   ⚠️ WARNING: Input appears to be None")
        
        # Step 2: Load finetuned checkpoint
        if isinstance(finetuned_checkpoint, str):
            if self.verbose:
                print(f"\n📂 STEP 2: Loading the Fine-tuned Model")
                print(f"   └─ This is the model AFTER training on a specific task")
                print(f"   └─ File: {finetuned_checkpoint}")
            finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu', weights_only=False)
            if self.verbose:
                if finetuned_state is not None:
                    print(f"   ✅ VALIDATION PASSED: File loaded successfully")
                else:
                    print(f"   ❌ VALIDATION FAILED: File loaded but returned None")
        else:
            if self.verbose:
                print(f"\n📂 STEP 2: Using Provided Fine-tuned State Dict/Model")
            finetuned_state = finetuned_checkpoint
            if self.verbose:
                if finetuned_state is not None:
                    if isinstance(finetuned_state, dict):
                        if len(finetuned_state) > 0:
                            print(f"   ✅ VALIDATION PASSED: State dict is valid and non-empty")
                        else:
                            print(f"   ⚠️ WARNING: State dict appears empty")
                    elif hasattr(finetuned_state, 'state_dict'):
                        print(f"   ✅ VALIDATION PASSED: Model object detected (will extract state dict)")
                    else:
                        print(f"   ✅ VALIDATION PASSED: Data received (will attempt extraction)")
                else:
                    print(f"   ⚠️ WARNING: Input appears to be None")
        
        # Step 3: Extract state dicts
        if self.verbose:
            print(f"\n🔍 STEP 3: Extracting Model Parameters")
            print(f"   └─ Converting checkpoints to parameter dictionaries...")
            print(f"   └─ (This handles various checkpoint formats: nested dicts, model objects, etc.)")
        
        pretrained_state = _extract_state_dict(pretrained_state)
        finetuned_state = _extract_state_dict(finetuned_state)
        
        if self.verbose:
            pretrained_count = len(pretrained_state)
            finetuned_count = len(finetuned_state)
            print(f"   ├─ Pretrained model: {pretrained_count:,} parameters")
            print(f"   └─ Fine-tuned model: {finetuned_count:,} parameters")
            
            # Validation: Check parameter counts match
            if pretrained_count == finetuned_count:
                print(f"   ✅ VALIDATION PASSED: Both models have same number of parameters")
            elif pretrained_count > 0 and finetuned_count > 0:
                overlap = len(set(pretrained_state.keys()) & set(finetuned_state.keys()))
                print(f"   ⚠️ WARNING: Parameter counts differ ({pretrained_count} vs {finetuned_count})")
                print(f"      └─ Overlapping parameters: {overlap:,} (will use these)")
            else:
                print(f"   ❌ VALIDATION FAILED: One or both models have no parameters!")
        
        # Step 4: Compute task vector (delta)
        if self.verbose:
            print(f"\n🧮 STEP 4: Computing the Task Vector (The Magic Step!)")
            print(f"   └─ For each parameter: delta = fine_tuned_value - pretrained_value")
            print(f"   └─ This tells us exactly what changed during training...")
        
        self.vector = {}
        skipped_count = 0
        skipped_reasons = {"missing": 0, "dtype": 0, "shape": 0}
        total_delta_norm = 0.0
        max_delta_norm = 0.0
        max_delta_param = ""
        zero_delta_count = 0
        
        for key in pretrained_state.keys():
            if key not in finetuned_state:
                skipped_count += 1
                skipped_reasons["missing"] += 1
                continue
            
            pretrained_param = pretrained_state[key]
            finetuned_param = finetuned_state[key]
            
            if skip_int64 and pretrained_param.dtype == torch.int64:
                skipped_count += 1
                skipped_reasons["dtype"] += 1
                continue
            if skip_uint8 and pretrained_param.dtype == torch.uint8:
                skipped_count += 1
                skipped_reasons["dtype"] += 1
                continue
            
            if pretrained_param.shape != finetuned_param.shape:
                skipped_count += 1
                skipped_reasons["shape"] += 1
                continue
            
            delta = finetuned_param - pretrained_param
            self.vector[key] = delta
            
            delta_norm = delta.norm().item()
            if delta_norm < 1e-10:
                zero_delta_count += 1
            total_delta_norm += delta_norm ** 2
            if delta_norm > max_delta_norm:
                max_delta_norm = delta_norm
                max_delta_param = key
        
        total_delta_norm = total_delta_norm ** 0.5
        
        if self.verbose:
            print(f"""
   📊 COMPUTATION RESULTS:
   ├─ Parameters processed: {len(self.vector):,}
   ├─ Parameters skipped: {skipped_count}
   │  ├─ Missing in fine-tuned: {skipped_reasons['missing']}
   │  ├─ Wrong dtype (int64/uint8): {skipped_reasons['dtype']}
   │  └─ Shape mismatch: {skipped_reasons['shape']}
   ├─ Total change magnitude (L2 norm): {total_delta_norm:.6f}
   ├─ Max single-param change: {max_delta_norm:.6f}
   │  └─ In parameter: {max_delta_param[:50]}{'...' if len(max_delta_param) > 50 else ''}
   └─ Parameters with zero change: {zero_delta_count}
""")
            
            # Validation checks
            print(f"   🔬 VALIDATION CHECKS:")
            
            # Check 1: Did we get any parameters?
            if len(self.vector) > 0:
                print(f"   ✅ CHECK 1 PASSED: Task vector has {len(self.vector):,} parameters")
            else:
                print(f"   ❌ CHECK 1 FAILED: No parameters in task vector!")
            
            # Check 2: Is there actual change?
            if total_delta_norm > 1e-10:
                print(f"   ✅ CHECK 2 PASSED: Model weights actually changed (norm={total_delta_norm:.6f})")
            else:
                print(f"   ❌ CHECK 2 FAILED: No weight changes detected! Models may be identical.")
            
            # Check 3: Are most parameters non-zero?
            non_zero_ratio = (len(self.vector) - zero_delta_count) / max(len(self.vector), 1)
            if non_zero_ratio > 0.5:
                print(f"   ✅ CHECK 3 PASSED: {non_zero_ratio*100:.1f}% of parameters have non-zero changes")
            elif non_zero_ratio > 0:
                print(f"   ⚠️ CHECK 3 WARNING: Only {non_zero_ratio*100:.1f}% of parameters changed (sparse update)")
            else:
                print(f"   ❌ CHECK 3 FAILED: All parameters have zero change!")
            
            # Check 4: Sanity check on magnitude
            if 1e-6 < total_delta_norm < 1e6:
                print(f"   ✅ CHECK 4 PASSED: Change magnitude is in reasonable range")
            elif total_delta_norm >= 1e6:
                print(f"   ⚠️ CHECK 4 WARNING: Very large changes detected - may indicate issue")
            else:
                print(f"   ⚠️ CHECK 4 WARNING: Very small changes - fine-tuning may have had minimal effect")
            
            # Verification: Can we reconstruct?
            print(f"""
   🧪 QUICK VERIFICATION TEST:
   └─ Testing: pretrained + task_vector ≈ fine_tuned ?""")
            
            # Pick a random parameter to verify
            if len(self.vector) > 0:
                test_key = list(self.vector.keys())[0]
                reconstructed = pretrained_state[test_key] + self.vector[test_key]
                original_finetuned = finetuned_state[test_key]
                reconstruction_error = (reconstructed - original_finetuned).abs().max().item()
                
                if reconstruction_error < 1e-5:
                    print(f"      ✅ VERIFIED: Reconstruction error = {reconstruction_error:.2e} (perfect!)")
                else:
                    print(f"      ❌ ERROR: Reconstruction error = {reconstruction_error:.2e} (unexpected!)")
            
            print(f"""
   💡 WHAT CAN YOU DO WITH THIS TASK VECTOR?
   • Add it to another model to transfer this task's knowledge
   • Combine with other task vectors for multi-task learning
   • Scale it (multiply by 0.5) to reduce the effect
   • Subtract it to "forget" what was learned
""")
            print(f"{'='*70}\n")
    
    def __add__(self, other: 'TaskVector') -> 'TaskVector':
        """
        Add two task vectors element-wise.
        
        Combines the "knowledge" from two task vectors. When applied to a base model,
        the resulting model will have capabilities from both tasks.
        
        Args:
            other: Another TaskVector to add
            
        Returns:
            New TaskVector with summed deltas for shared parameters
            
        Example:
            >>> tv_combined = tv_task1 + tv_task2
        """
        # Create new TaskVector without calling __init__ (avoid checkpoint loading)
        result = TaskVector.__new__(TaskVector)
        
        # Combine task names for debugging
        result.task_name = f"{self.task_name}+{other.task_name}" if self.task_name and other.task_name else None
        result.vector = {}
        
        # Add deltas for parameters that exist in both task vectors
        for key in self.vector.keys():
            if key in other.vector:
                result.vector[key] = self.vector[key] + other.vector[key]
        
        return result
    
    def __sub__(self, other: 'TaskVector') -> 'TaskVector':
        """
        Subtract one task vector from another.
        
        Can be used to "remove" capabilities or compute the difference
        between what two models learned on different tasks.
        
        Args:
            other: TaskVector to subtract
            
        Returns:
            New TaskVector with delta differences for shared parameters
            
        Example:
            >>> tv_diff = tv_general - tv_specific
        """
        result = TaskVector.__new__(TaskVector)
        result.task_name = f"{self.task_name}-{other.task_name}" if self.task_name and other.task_name else None
        result.vector = {}
        
        for key in self.vector.keys():
            if key in other.vector:
                result.vector[key] = self.vector[key] - other.vector[key]
        
        return result
    
    def __mul__(self, scalar: float) -> 'TaskVector':
        """
        Multiply task vector by a scalar.
        
        Scales all deltas by the given factor. Useful for:
        - Weighting tasks differently in a merge (e.g., tv1 * 0.7 + tv2 * 0.3)
        - Regularization (reduce magnitude of changes)
        - Amplification (increase effect of fine-tuning)
        
        Args:
            scalar: Multiplier for all deltas
            
        Returns:
            New TaskVector with scaled deltas
            
        Example:
            >>> tv_scaled = tv * 0.5  # Half the effect
        """
        result = TaskVector.__new__(TaskVector)
        result.task_name = self.task_name
        result.vector = {key: val * scalar for key, val in self.vector.items()}
        return result
    
    def __rmul__(self, scalar: float) -> 'TaskVector':
        """
        Right multiply task vector by scalar (enables 0.5 * tv syntax).
        
        This is called when scalar is on the left: 0.5 * tv
        Delegates to __mul__ for the actual computation.
        """
        return self.__mul__(scalar)
    
    def apply_to(self, pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]], verbose: bool = None) -> Dict[str, torch.Tensor]:
        """
        Apply task vector to pretrained model to get finetuned model.
        
        Reconstructs the fine-tuned model by adding the task vector (deltas)
        back to the pretrained model weights:
        
            finetuned_param = pretrained_param + delta
        
        For parameters not in the task vector, the pretrained values are kept.
        
        Args:
            pretrained_checkpoint: Path to checkpoint or state dict of pretrained model
            verbose: Override instance verbose setting for this call
            
        Returns:
            State dict with task vector applied (approximates the fine-tuned model)
            
        Example:
            >>> # Reconstruct finetuned model
            >>> tv = TaskVector("base.pt", "finetuned.pt")
            >>> reconstructed = tv.apply_to("base.pt")
            >>> # reconstructed ≈ finetuned model's state dict
            
            >>> # Create merged model from multiple tasks
            >>> tv_merged = tv1 * 0.5 + tv2 * 0.5
            >>> merged_model = tv_merged.apply_to("base.pt")
        """
        # Use instance verbose setting if not overridden
        _verbose = verbose if verbose is not None else getattr(self, 'verbose', False)
        
        if _verbose:
            task_str = f" ({self.task_name})" if self.task_name else ""
            print(f"\n{'='*70}")
            print(f"📚 TUTORIAL: Applying Task Vector{task_str} to Base Model")
            print(f"{'='*70}")
            print(f"""
🎯 WHAT DOES "APPLYING" A TASK VECTOR MEAN?
   We're reconstructing a model by adding the learned changes back:
   
   new_model = base_model + task_vector
   
   This creates a model that has the capabilities of the task(s)
   represented by the task vector.
""")
        
        # Load pretrained checkpoint
        if isinstance(pretrained_checkpoint, str):
            if _verbose:
                print(f"📂 STEP 1: Loading Base Model")
                print(f"   └─ File: {pretrained_checkpoint}")
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
            if _verbose:
                print(f"   ✅ LOADED: File loaded successfully")
        else:
            if _verbose:
                print(f"📂 STEP 1: Using Provided Base Model State Dict")
            pretrained_state = pretrained_checkpoint
        
        # Extract state dict from various checkpoint formats including torch.nn.Module objects
        pretrained_state = _extract_state_dict(pretrained_state)
        
        if _verbose:
            print(f"   └─ Base model has {len(pretrained_state):,} parameters")
        
        # Make a copy to avoid modifying the input (only clone tensors)
        if _verbose:
            print(f"\n🔄 STEP 2: Creating Copy of Base Model")
            print(f"   └─ (We don't want to modify the original)")
        pretrained_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in pretrained_state.items()}
        if _verbose:
            print(f"   ✅ DONE: Safe copy created")
        
        # Apply task vector to each parameter
        if _verbose:
            print(f"\n🧮 STEP 3: Applying Task Vector (Adding Deltas)")
            print(f"   └─ For each parameter: new_value = base_value + delta")
        
        result_state = {}
        applied_count = 0
        unchanged_count = 0
        total_change_norm = 0.0
        
        for key in pretrained_state.keys():
            if key in self.vector:
                # Add delta to pretrained: finetuned = pretrained + delta
                result_state[key] = pretrained_state[key] + self.vector[key]
                applied_count += 1
                total_change_norm += self.vector[key].norm().item() ** 2
            else:
                # Keep pretrained value for parameters not in task vector
                result_state[key] = pretrained_state[key]
                unchanged_count += 1
        
        total_change_norm = total_change_norm ** 0.5
        
        if _verbose:
            print(f"""
   📊 APPLICATION RESULTS:
   ├─ Parameters modified: {applied_count:,}
   ├─ Parameters unchanged: {unchanged_count:,}
   └─ Total modification magnitude: {total_change_norm:.6f}
""")
            
            # Validation checks
            print(f"   🔬 VALIDATION CHECKS:")
            
            # Check 1: All parameters accounted for
            total_params = applied_count + unchanged_count
            if total_params == len(pretrained_state):
                print(f"   ✅ CHECK 1 PASSED: All {total_params:,} parameters accounted for")
            else:
                print(f"   ❌ CHECK 1 FAILED: Parameter count mismatch!")
            
            # Check 2: Output has correct structure
            if len(result_state) == len(pretrained_state):
                print(f"   ✅ CHECK 2 PASSED: Output has same number of parameters as input")
            else:
                print(f"   ❌ CHECK 2 FAILED: Output parameter count differs from input!")
            
            # Check 3: At least some parameters were modified
            if applied_count > 0:
                print(f"   ✅ CHECK 3 PASSED: {applied_count:,} parameters were updated")
            else:
                print(f"   ⚠️ CHECK 3 WARNING: No parameters were modified (empty task vector?)")
            
            # Verification test
            if applied_count > 0:
                print(f"""
   🧪 VERIFICATION TEST:
   └─ Checking: result = base + delta ?""")
                test_key = list(self.vector.keys())[0]
                expected = pretrained_state[test_key] + self.vector[test_key]
                actual = result_state[test_key]
                verification_error = (expected - actual).abs().max().item()
                
                if verification_error < 1e-6:
                    print(f"      ✅ VERIFIED: Computation is correct (error={verification_error:.2e})")
                else:
                    print(f"      ❌ ERROR: Unexpected computation error={verification_error:.2e}")
            
            print(f"""
   💡 WHAT'S NEXT?
   • Load this state dict into your model: model.load_state_dict(result)
   • Run inference or evaluation to test the merged model
   • Compare performance against the original fine-tuned models
""")
            print(f"{'='*70}\n")
        
        return result_state


class QuantizedTaskVector:
    """
    Task vector with quantized deltas.
    
    Instead of storing full-precision (FP32) deltas, this class stores quantized
    deltas using low-bit integers (e.g., int8, uint8). This reduces storage
    requirements at the cost of some precision.
    
    === WHY QUANTIZE TASK VECTORS? ===
    
    - **Storage Efficiency**: 8-bit quantization = 4x compression vs FP32
    - **Bandwidth**: Faster transfer and loading
    - **Memory**: Fit more task vectors in GPU/CPU memory
    
    === HOW IT WORKS ===
    
    1. Receive pre-quantized deltas with their quantization parameters
    2. Store quantized indices (int8/uint8) + scale/zero_point metadata
    3. On demand, dequantize to reconstruct approximate deltas
    4. Apply dequantized deltas to base model
    
    === USAGE ===
    
        # First, quantize a task vector
        >>> from quantization_utils import asymmetric_quantization
        >>> tv = TaskVector("base.pt", "finetuned.pt")
        >>> quantized_deltas = {}
        >>> for key, delta in tv.vector.items():
        ...     q, scale, zp = asymmetric_quantization(delta, qbit=8)
        ...     quantized_deltas[key] = {"quantized": q, "scale": scale, "zero_point": zp}
        
        # Create quantized task vector
        >>> qtv = QuantizedTaskVector(quantized_deltas, method="asymmetric")
        
        # Dequantize when needed
        >>> restored_deltas = qtv.dequantize()
        
        # Or directly apply to base model
        >>> merged_state = qtv.apply_to("base.pt")
    
    Attributes:
        quantized_deltas: Dict mapping param_name -> {quantized, scale, [zero_point]}
        method: Quantization method ("asymmetric" or "absmax")
    """
    
    def __init__(
        self,
        quantized_deltas: Dict[str, Dict],
        method: str = "asymmetric"
    ):
        """
        Initialize from quantized deltas.
        
        Args:
            quantized_deltas: Dictionary mapping parameter name -> quantization payload
                             Payload should contain: 'quantized', 'scale', and optionally 'zero_point'
            method: Quantization method ("asymmetric" or "absmax")
            
        Example payload format:
            {
                "layer1.weight": {
                    "quantized": tensor([...], dtype=uint8),
                    "scale": tensor(0.00123),
                    "zero_point": tensor(128.0)  # Only for asymmetric
                },
                ...
            }
        """
        self.quantized_deltas = quantized_deltas
        self.method = method
    
    def dequantize(self) -> Dict[str, torch.Tensor]:
        """
        Dequantize to get task vector.
        
        Reconstructs approximate delta tensors from the stored quantized values.
        The reconstruction is not exact due to quantization error.
        
        Returns:
            Dictionary mapping parameter name -> dequantized delta tensor (FP32)
            
        Example:
            >>> qtv = QuantizedTaskVector(quantized_deltas, method="asymmetric")
            >>> deltas = qtv.dequantize()
            >>> print(deltas["layer1.weight"].dtype)  # torch.float32
        """
        vector = {}
        
        for key, payload in self.quantized_deltas.items():
            # Extract quantized data and parameters
            X_q = payload['quantized']
            scale = payload['scale']
            
            # Dequantize based on method
            if self.method == "asymmetric":
                zero_point = payload.get('zero_point', torch.tensor(0.0))
                delta = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
            else:  # absmax
                delta = quantization_utils.dequantize_absmax(X_q, scale)
            
            vector[key] = delta
        
        return vector
    
    def apply_to(self, pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Apply quantized task vector to pretrained model.
        
        Dequantizes the stored deltas and adds them to the pretrained model,
        producing an approximate reconstruction of the fine-tuned model.
        
        Args:
            pretrained_checkpoint: Path or state dict of pretrained model
            
        Returns:
            State dict with dequantized task vector applied
            
        Example:
            >>> qtv = QuantizedTaskVector(quantized_deltas)
            >>> merged = qtv.apply_to("base_model.pt")
            >>> torch.save(merged, "merged_model.pt")
        """
        # Load pretrained model
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        else:
            pretrained_state = pretrained_checkpoint
        
        # Extract state dict from various checkpoint formats including torch.nn.Module objects
        pretrained_state = _extract_state_dict(pretrained_state)
        
        # Dequantize the task vector
        vector = self.dequantize()
        
        # Apply dequantized deltas to pretrained model
        result_state = {}
        for key in pretrained_state.keys():
            if key in vector:
                result_state[key] = pretrained_state[key] + vector[key]
            else:
                result_state[key] = pretrained_state[key]
        
        return result_state


class QuantizedFinetunedModel:
    """
    Stores finetuned model by quantizing weights directly.
    
    Reconstructs finetuned weights, then computes task vector as (finetuned - pretrained).
    """
    
    def __init__(
        self,
        finetuned_checkpoint: Union[str, Dict[str, torch.Tensor]],
        qbit: int = 8,
        method: str = "asymmetric",
        skip_int64: bool = True,
        skip_uint8: bool = True
    ):
        """
        Initialize by quantizing finetuned model.
        
        Args:
            finetuned_checkpoint: Finetuned model checkpoint
            qbit: Number of quantization bits
            method: Quantization method
            skip_int64: Skip int64 parameters
            skip_uint8: Skip uint8 parameters
        """
        if isinstance(finetuned_checkpoint, str):
            finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu', weights_only=False)
        else:
            finetuned_state = finetuned_checkpoint
        
        # Extract state dict from various checkpoint formats including torch.nn.Module objects
        finetuned_state = _extract_state_dict(finetuned_state)
        
        self.quantized_weights = {}
        self.qbit = qbit
        self.method = method
        
        for key, param in finetuned_state.items():
            # Skip certain dtypes
            if skip_int64 and param.dtype == torch.int64:
                continue
            if skip_uint8 and param.dtype == torch.uint8:
                continue
            
            # Quantize
            if method == "asymmetric":
                X_q, scale, zero_point = quantization_utils.asymmetric_quantization(param, qbit)
                self.quantized_weights[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'zero_point': zero_point,
                    'shape': param.shape
                }
            else:  # absmax
                X_q, scale = quantization_utils.absmax_quantization(param, qbit)
                self.quantized_weights[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'shape': param.shape
                }
    
    def dequantize(self) -> Dict[str, torch.Tensor]:
        """
        Dequantize to reconstruct finetuned model.
        
        Returns:
            State dict of finetuned model
        """
        finetuned_state = {}
        
        for key, payload in self.quantized_weights.items():
            X_q = payload['quantized']
            scale = payload['scale']
            shape = payload['shape']
            
            if self.method == "asymmetric":
                zero_point = payload['zero_point']
                param = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
            else:  # absmax
                param = quantization_utils.dequantize_absmax(X_q, scale)
            
            finetuned_state[key] = param.reshape(shape)
        
        return finetuned_state
    
    def get_task_vector(self, pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Get task vector by subtracting pretrained from dequantized finetuned.
        
        Args:
            pretrained_checkpoint: Pretrained model checkpoint
            
        Returns:
            Task vector (delta) dictionary
        """
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        else:
            pretrained_state = pretrained_checkpoint
        
        # Extract state dict from various checkpoint formats including torch.nn.Module objects
        pretrained_state = _extract_state_dict(pretrained_state)
        
        finetuned_state = self.dequantize()
        
        task_vector = {}
        for key in finetuned_state.keys():
            if key in pretrained_state:
                task_vector[key] = finetuned_state[key] - pretrained_state[key]
        
        return task_vector


class QuantizedBaseAndTaskVector:
    """
    Stores both quantized base model and quantized task vectors.
    
    Enables residual representation: quantized_base + quantized_task_vector.
    """
    
    def __init__(
        self,
        pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]],
        task_vector: Union[TaskVector, Dict[str, torch.Tensor]],
        base_qbit: int = 8,
        task_qbit: int = 8,
        method: str = "asymmetric",
        skip_int64: bool = True,
        skip_uint8: bool = True
    ):
        """
        Initialize by quantizing base and task vector separately.
        
        Args:
            pretrained_checkpoint: Pretrained model checkpoint
            task_vector: TaskVector object or delta dictionary
            base_qbit: Bits for base model quantization
            task_qbit: Bits for task vector quantization
            method: Quantization method
            skip_int64: Skip int64 parameters
            skip_uint8: Skip uint8 parameters
        """
        # Load pretrained
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        else:
            pretrained_state = pretrained_checkpoint
        
        # Extract state dict from various checkpoint formats including torch.nn.Module objects
        pretrained_state = _extract_state_dict(pretrained_state)
        
        # Get task vector dict
        if isinstance(task_vector, TaskVector):
            task_vector_dict = task_vector.vector
        else:
            task_vector_dict = task_vector
        
        self.method = method
        self.base_qbit = base_qbit
        self.task_qbit = task_qbit
        
        # Quantize base model
        self.quantized_base = {}
        for key, param in pretrained_state.items():
            if skip_int64 and param.dtype == torch.int64:
                continue
            if skip_uint8 and param.dtype == torch.uint8:
                continue
            
            if method == "asymmetric":
                X_q, scale, zero_point = quantization_utils.asymmetric_quantization(param, base_qbit)
                self.quantized_base[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'zero_point': zero_point,
                    'shape': param.shape
                }
            else:
                X_q, scale = quantization_utils.absmax_quantization(param, base_qbit)
                self.quantized_base[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'shape': param.shape
                }
        
        # Quantize task vector
        self.quantized_task = {}
        for key, delta in task_vector_dict.items():
            if method == "asymmetric":
                X_q, scale, zero_point = quantization_utils.asymmetric_quantization(delta, task_qbit)
                self.quantized_task[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'zero_point': zero_point,
                    'shape': delta.shape
                }
            else:
                X_q, scale = quantization_utils.absmax_quantization(delta, task_qbit)
                self.quantized_task[key] = {
                    'quantized': X_q,
                    'scale': scale,
                    'shape': delta.shape
                }
    
    def dequantize(self) -> Dict[str, torch.Tensor]:
        """
        Dequantize and merge base + task vector.
        
        Returns:
            State dict of finetuned model
        """
        result_state = {}
        
        # Dequantize base
        for key, payload in self.quantized_base.items():
            X_q = payload['quantized']
            scale = payload['scale']
            shape = payload['shape']
            
            if self.method == "asymmetric":
                zero_point = payload['zero_point']
                param = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
            else:
                param = quantization_utils.dequantize_absmax(X_q, scale)
            
            result_state[key] = param.reshape(shape)
        
        # Add task vector
        for key, payload in self.quantized_task.items():
            X_q = payload['quantized']
            scale = payload['scale']
            shape = payload['shape']
            
            if self.method == "asymmetric":
                zero_point = payload['zero_point']
                delta = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
            else:
                delta = quantization_utils.dequantize_absmax(X_q, scale)
            
            delta = delta.reshape(shape)
            
            if key in result_state:
                result_state[key] = result_state[key] + delta
            else:
                result_state[key] = delta
        
        return result_state
    

