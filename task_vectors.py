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
        skip_uint8: bool = True
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
            
        Example:
            # From file paths
            >>> tv = TaskVector("./base_model.pt", "./finetuned_on_cars.pt", task_name="Cars")
            
            # From already-loaded state dicts
            >>> tv = TaskVector(base_state_dict, finetuned_state_dict)
        """
        self.task_name = task_name
        
        # Step 1: Load pretrained checkpoint
        # Can accept either a file path (str) or an already-loaded state dict
        if isinstance(pretrained_checkpoint, str):
            # Load from file, map to CPU to avoid GPU memory issues with large models
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint
        
        # Step 2: Load finetuned checkpoint
        if isinstance(finetuned_checkpoint, str):
            finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu')
        else:
            finetuned_state = finetuned_checkpoint
        
        # Step 3: Handle nested state dicts
        # Some checkpoint formats wrap the state dict in a dictionary with a "state_dict" key
        # (e.g., PyTorch Lightning checkpoints)
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        if "state_dict" in finetuned_state:
            finetuned_state = finetuned_state["state_dict"]
        
        # Step 4: Compute task vector (delta) for each parameter
        self.vector = {}
        for key in pretrained_state.keys():
            # Skip parameters that don't exist in the finetuned model
            # (indicates architecture mismatch or intentionally frozen layers)
            if key not in finetuned_state:
                continue
            
            pretrained_param = pretrained_state[key]
            finetuned_param = finetuned_state[key]
            
            # Skip certain dtypes that typically aren't learnable parameters
            # int64: Often used for buffer indices, position embeddings, etc.
            # uint8: Often used for quantized weights or mask tensors
            if skip_int64 and pretrained_param.dtype == torch.int64:
                continue
            if skip_uint8 and pretrained_param.dtype == torch.uint8:
                continue
            
            # Skip if shapes don't match (indicates incompatible architectures)
            if pretrained_param.shape != finetuned_param.shape:
                continue
            
            # Compute the delta: what changed during fine-tuning
            delta = finetuned_param - pretrained_param
            self.vector[key] = delta
    
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
    
    def apply_to(self, pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Apply task vector to pretrained model to get finetuned model.
        
        Reconstructs the fine-tuned model by adding the task vector (deltas)
        back to the pretrained model weights:
        
            finetuned_param = pretrained_param + delta
        
        For parameters not in the task vector, the pretrained values are kept.
        
        Args:
            pretrained_checkpoint: Path to checkpoint or state dict of pretrained model
            
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
        # Load pretrained checkpoint
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            # Make a copy to avoid modifying the input
            pretrained_state = pretrained_checkpoint.copy()
        
        # Handle nested state dicts
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
        # Apply task vector to each parameter
        result_state = {}
        for key in pretrained_state.keys():
            if key in self.vector:
                # Add delta to pretrained: finetuned = pretrained + delta
                result_state[key] = pretrained_state[key] + self.vector[key]
            else:
                # Keep pretrained value for parameters not in task vector
                result_state[key] = pretrained_state[key]
        
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
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint.copy()
        
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
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
            finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu')
        else:
            finetuned_state = finetuned_checkpoint
        
        if "state_dict" in finetuned_state:
            finetuned_state = finetuned_state["state_dict"]
        
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
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint
        
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
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
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint
        
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
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
    

