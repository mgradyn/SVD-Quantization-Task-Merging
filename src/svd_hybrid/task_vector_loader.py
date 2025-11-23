"""
Task vector loading utilities extended from TVQ logic.
Loads fine-tuned model checkpoints and computes task vectors (deltas from base model).
"""
import torch
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        State dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        else:
            # Assume it's already a state dict
            return checkpoint
    
    return checkpoint


def compute_task_vector(
    base_state: Dict[str, torch.Tensor],
    finetuned_state: Dict[str, torch.Tensor],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Compute task vector (delta) between fine-tuned and base model.
    
    Args:
        base_state: Base model state dict
        finetuned_state: Fine-tuned model state dict
        device: Device for computation
        
    Returns:
        Dictionary mapping parameter name -> delta tensor
    """
    task_vector = {}
    
    for key in base_state.keys():
        if key not in finetuned_state:
            continue
        
        base_param = base_state[key].to(device)
        finetuned_param = finetuned_state[key].to(device)
        
        if base_param.shape != finetuned_param.shape:
            print(f"Warning: Shape mismatch for {key}, skipping")
            continue
        
        delta = finetuned_param - base_param
        task_vector[key] = delta.detach()
    
    return task_vector


def load_task_vectors(
    base_model_path: str,
    task_checkpoint_paths: Dict[str, str],
    device: str = "cpu",
    filter_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load task vectors for multiple tasks.
    
    Args:
        base_model_path: Path to base model checkpoint
        task_checkpoint_paths: Dictionary mapping task_name -> checkpoint_path
        device: Device for computation
        filter_keys: Optional list of parameter name patterns to include
        
    Returns:
        Dictionary mapping task_name -> parameter_name -> delta tensor
    """
    # Load base model
    print(f"Loading base model from {base_model_path}")
    base_state = load_checkpoint(base_model_path, device)
    
    # Filter base state if needed
    if filter_keys is not None:
        base_state = {k: v for k, v in base_state.items() 
                     if any(pattern in k for pattern in filter_keys)}
    
    task_vectors = {}
    
    for task_name, ckpt_path in task_checkpoint_paths.items():
        print(f"Loading task vector for {task_name} from {ckpt_path}")
        finetuned_state = load_checkpoint(ckpt_path, device)
        
        # Filter fine-tuned state
        if filter_keys is not None:
            finetuned_state = {k: v for k, v in finetuned_state.items() 
                             if any(pattern in k for pattern in filter_keys)}
        
        task_vector = compute_task_vector(base_state, finetuned_state, device)
        task_vectors[task_name] = task_vector
    
    return task_vectors


def get_parameter_names(task_vectors: Dict[str, Dict[str, torch.Tensor]]) -> List[str]:
    """
    Get all unique parameter names across all tasks.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        
    Returns:
        Sorted list of parameter names
    """
    param_names = set()
    for task_vector in task_vectors.values():
        param_names.update(task_vector.keys())
    
    return sorted(list(param_names))


def organize_by_parameter(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Reorganize task vectors by parameter instead of by task.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        
    Returns:
        Dictionary mapping parameter_name -> task_name -> delta tensor
    """
    by_param = {}
    
    for task_name, task_vector in task_vectors.items():
        for param_name, delta in task_vector.items():
            if param_name not in by_param:
                by_param[param_name] = {}
            by_param[param_name][task_name] = delta
    
    return by_param


def flatten_task_deltas(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    param_name: str
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Get flattened delta vectors for a specific parameter across all tasks.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        param_name: Parameter name to extract
        
    Returns:
        Tuple of (list of flattened delta tensors, list of task names in same order)
    """
    deltas = []
    task_names = []
    
    for task_name, task_vector in task_vectors.items():
        if param_name in task_vector:
            delta = task_vector[param_name]
            deltas.append(delta.flatten())
            task_names.append(task_name)
    
    return deltas, task_names


def get_task_checkpoint_paths(
    checkpoint_dir: str,
    task_names: List[str]
) -> Dict[str, str]:
    """
    Construct checkpoint paths for tasks.
    
    Args:
        checkpoint_dir: Base directory containing checkpoints
        task_names: List of task identifiers
        
    Returns:
        Dictionary mapping task_name -> checkpoint_path
    """
    checkpoint_paths = {}
    
    for task_name in task_names:
        # Try various naming conventions
        possible_paths = [
            os.path.join(checkpoint_dir, f"{task_name}.pt"),
            os.path.join(checkpoint_dir, f"{task_name}.pth"),
            os.path.join(checkpoint_dir, task_name, "checkpoint.pt"),
            os.path.join(checkpoint_dir, task_name, "model.pt"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_paths[task_name] = path
                break
        else:
            raise FileNotFoundError(f"No checkpoint found for task {task_name} in {checkpoint_dir}")
    
    return checkpoint_paths
