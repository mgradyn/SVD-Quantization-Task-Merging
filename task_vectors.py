"""
Task Vector classes following TVQ reference implementation.

Provides utilities for working with task vectors (parameter deltas between
fine-tuned and pretrained models) and quantized representations.
"""
import torch
from typing import Dict, Optional, Union
from pathlib import Path
import quantization_utils


class TaskVector:
    """
    Represents a task vector (delta = finetuned - pretrained).
    
    Attributes:
        vector: Dictionary mapping parameter name -> delta tensor
        task_name: Optional task identifier
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
        
        Args:
            pretrained_checkpoint: Path or state dict of pretrained model
            finetuned_checkpoint: Path or state dict of finetuned model
            task_name: Optional task identifier
            skip_int64: Skip int64 parameters (e.g., buffers)
            skip_uint8: Skip uint8 parameters
        """
        self.task_name = task_name
        
        # Load checkpoints
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint
            
        if isinstance(finetuned_checkpoint, str):
            finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu')
        else:
            finetuned_state = finetuned_checkpoint
        
        # Handle nested state dicts
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        if "state_dict" in finetuned_state:
            finetuned_state = finetuned_state["state_dict"]
        
        # Compute task vector
        self.vector = {}
        for key in pretrained_state.keys():
            if key not in finetuned_state:
                continue
            
            pretrained_param = pretrained_state[key]
            finetuned_param = finetuned_state[key]
            
            # Skip certain dtypes following TVQ reference
            if skip_int64 and pretrained_param.dtype == torch.int64:
                continue
            if skip_uint8 and pretrained_param.dtype == torch.uint8:
                continue
            
            if pretrained_param.shape != finetuned_param.shape:
                continue
            
            # Compute delta
            delta = finetuned_param - pretrained_param
            self.vector[key] = delta
    
    def __add__(self, other: 'TaskVector') -> 'TaskVector':
        """Add two task vectors."""
        result = TaskVector.__new__(TaskVector)
        result.task_name = f"{self.task_name}+{other.task_name}" if self.task_name and other.task_name else None
        result.vector = {}
        
        for key in self.vector.keys():
            if key in other.vector:
                result.vector[key] = self.vector[key] + other.vector[key]
        
        return result
    
    def __sub__(self, other: 'TaskVector') -> 'TaskVector':
        """Subtract two task vectors."""
        result = TaskVector.__new__(TaskVector)
        result.task_name = f"{self.task_name}-{other.task_name}" if self.task_name and other.task_name else None
        result.vector = {}
        
        for key in self.vector.keys():
            if key in other.vector:
                result.vector[key] = self.vector[key] - other.vector[key]
        
        return result
    
    def __mul__(self, scalar: float) -> 'TaskVector':
        """Multiply task vector by scalar."""
        result = TaskVector.__new__(TaskVector)
        result.task_name = self.task_name
        result.vector = {key: val * scalar for key, val in self.vector.items()}
        return result
    
    def __rmul__(self, scalar: float) -> 'TaskVector':
        """Right multiply task vector by scalar."""
        return self.__mul__(scalar)
    
    def apply_to(self, pretrained_checkpoint: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Apply task vector to pretrained model to get finetuned model.
        
        Args:
            pretrained_checkpoint: Pretrained model checkpoint
            
        Returns:
            State dict of model with task vector applied
        """
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint.copy()
        
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
        result_state = {}
        for key in pretrained_state.keys():
            if key in self.vector:
                result_state[key] = pretrained_state[key] + self.vector[key]
            else:
                result_state[key] = pretrained_state[key]
        
        return result_state


class QuantizedTaskVector:
    """
    Task vector with quantized deltas.
    
    Reconstructs task vector from stored quantized deltas.
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
        """
        self.quantized_deltas = quantized_deltas
        self.method = method
    
    def dequantize(self) -> Dict[str, torch.Tensor]:
        """
        Dequantize to get task vector.
        
        Returns:
            Dictionary mapping parameter name -> delta tensor
        """
        vector = {}
        
        for key, payload in self.quantized_deltas.items():
            X_q = payload['quantized']
            scale = payload['scale']
            
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
        
        Args:
            pretrained_checkpoint: Pretrained model checkpoint
            
        Returns:
            State dict with dequantized task vector applied
        """
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            pretrained_state = pretrained_checkpoint.copy()
        
        if "state_dict" in pretrained_state:
            pretrained_state = pretrained_state["state_dict"]
        
        vector = self.dequantize()
        
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
    

