"""
Tall Masks binary mask loading with support for union, intersection, majority voting.
"""
import torch
import os
from typing import Dict, List, Optional
from pathlib import Path


def load_single_mask(mask_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load a single tall_masks binary mask file.
    
    Args:
        mask_path: Path to mask file (.pt or .pth)
        device: Device to load mask to
        
    Returns:
        Dictionary mapping parameter names to binary masks
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask_dict = torch.load(mask_path, map_location=device)
    
    # Ensure all masks are boolean or can be converted
    for key, mask in mask_dict.items():
        if mask.dtype != torch.bool:
            mask_dict[key] = mask.bool()
    
    return mask_dict


def load_task_masks(mask_dir: str, task_names: List[str], device: str = "cpu") -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load masks for multiple tasks.
    
    Args:
        mask_dir: Directory containing mask files
        task_names: List of task identifiers
        device: Device to load masks to
        
    Returns:
        Dictionary mapping task_name -> parameter_name -> mask
    """
    task_masks = {}
    
    for task_name in task_names:
        # Try various naming conventions
        possible_paths = [
            os.path.join(mask_dir, f"{task_name}_mask.pt"),
            os.path.join(mask_dir, f"{task_name}.pt"),
            os.path.join(mask_dir, task_name, "mask.pt"),
        ]
        
        mask_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mask_path = path
                break
        
        if mask_path is None:
            print(f"Warning: No mask found for task {task_name}, using all-ones mask")
            task_masks[task_name] = None  # Will be handled as full mask
        else:
            task_masks[task_name] = load_single_mask(mask_path, device)
    
    return task_masks


def compute_union_mask(masks: List[torch.Tensor]) -> torch.Tensor:
    """Compute union of binary masks (OR operation)."""
    if not masks:
        raise ValueError("Empty mask list")
    
    result = masks[0].clone()
    for mask in masks[1:]:
        result = result | mask
    
    return result


def compute_intersection_mask(masks: List[torch.Tensor]) -> torch.Tensor:
    """Compute intersection of binary masks (AND operation)."""
    if not masks:
        raise ValueError("Empty mask list")
    
    result = masks[0].clone()
    for mask in masks[1:]:
        result = result & mask
    
    return result


def compute_majority_mask(masks: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
    """
    Compute majority voting mask.
    
    Args:
        masks: List of binary masks
        threshold: Fraction of masks that must be True for output to be True
        
    Returns:
        Binary mask where True if >= threshold fraction of input masks are True
    """
    if not masks:
        raise ValueError("Empty mask list")
    
    # Stack and sum
    stacked = torch.stack([m.float() for m in masks], dim=0)
    vote_sum = stacked.sum(dim=0)
    
    # Apply threshold
    result = vote_sum >= (threshold * len(masks))
    
    return result


def combine_masks(
    task_masks: Dict[str, Dict[str, torch.Tensor]],
    strategy: str = "union",
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Combine masks from multiple tasks using specified strategy.
    
    Args:
        task_masks: Dictionary mapping task_name -> parameter_name -> mask
        strategy: Combination strategy ("union", "intersection", "majority")
        device: Device for computation
        
    Returns:
        Dictionary mapping parameter_name -> combined mask
    """
    if not task_masks:
        return {}
    
    # Get all parameter names across all tasks
    all_param_names = set()
    for task_name, param_masks in task_masks.items():
        if param_masks is not None:
            all_param_names.update(param_masks.keys())
    
    combined = {}
    
    for param_name in all_param_names:
        # Collect masks for this parameter across tasks
        param_mask_list = []
        
        for task_name, param_masks in task_masks.items():
            if param_masks is None:
                # Task has no mask file, skip or use full mask
                continue
            
            if param_name in param_masks:
                param_mask_list.append(param_masks[param_name].to(device))
        
        if not param_mask_list:
            continue
        
        # Combine based on strategy
        if strategy == "union":
            combined[param_name] = compute_union_mask(param_mask_list)
        elif strategy == "intersection":
            combined[param_name] = compute_intersection_mask(param_mask_list)
        elif strategy == "majority":
            combined[param_name] = compute_majority_mask(param_mask_list)
        else:
            raise ValueError(f"Unknown mask strategy: {strategy}")
    
    return combined


def apply_mask_to_tensor(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply binary mask to tensor, returning masked (signal) portion.
    
    Args:
        tensor: Input tensor (any shape)
        mask: Binary mask (same shape as tensor)
        
    Returns:
        1D tensor containing values where mask is True
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tensor {tensor.shape} vs mask {mask.shape}")
    
    flat_tensor = tensor.flatten()
    flat_mask = mask.flatten()
    
    return flat_tensor[flat_mask]


def get_unmasked_portion(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Get unmasked (noise) portion of tensor.
    
    Args:
        tensor: Input tensor (any shape)
        mask: Binary mask (same shape as tensor)
        
    Returns:
        1D tensor containing values where mask is False
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tensor {tensor.shape} vs mask {mask.shape}")
    
    flat_tensor = tensor.flatten()
    flat_mask = mask.flatten()
    
    return flat_tensor[~flat_mask]


def reconstruct_from_masked(
    masked_values: torch.Tensor,
    unmasked_values: Optional[torch.Tensor],
    mask: torch.Tensor,
    original_shape: torch.Size
) -> torch.Tensor:
    """
    Reconstruct full tensor from masked and optionally unmasked portions.
    
    Args:
        masked_values: Values for masked (signal) region
        unmasked_values: Values for unmasked (noise) region (optional)
        mask: Binary mask
        original_shape: Original tensor shape
        
    Returns:
        Reconstructed tensor with original shape
    """
    flat_mask = mask.flatten()
    result_flat = torch.zeros_like(flat_mask, dtype=masked_values.dtype, device=masked_values.device)
    
    # Fill in masked portion
    result_flat[flat_mask] = masked_values
    
    # Fill in unmasked portion if provided
    if unmasked_values is not None:
        result_flat[~flat_mask] = unmasked_values
    
    return result_flat.view(original_shape)
