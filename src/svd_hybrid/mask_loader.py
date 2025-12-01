"""
Tall Masks binary mask loading with support for union, intersection, majority voting.

=== TUTORIAL: Understanding Tall Masks ===

Tall Masks are binary masks that identify which parameters (weights) are important
for each task. They were introduced in the "Tall Masks" paper to improve model
merging by focusing on task-specific parameters.

=== WHAT ARE TALL MASKS? ===

For each task, a Tall Mask is a binary tensor (True/False) with the same shape
as the model parameters. True indicates the parameter is important for that task.

Example:
    mask["layer1.weight"] = tensor([[True, False, True, ...], ...])

=== MASK COMBINATION STRATEGIES ===

When merging multiple tasks, we need to combine their masks:

1. **Union (OR)**: A parameter is important if ANY task uses it
   - Most inclusive - uses all potentially useful parameters
   - Good when tasks are diverse
   - result = mask1 | mask2 | mask3 | ...

2. **Intersection (AND)**: A parameter is important if ALL tasks use it
   - Most selective - only parameters important to all tasks
   - Good when tasks share a lot of structure
   - result = mask1 & mask2 & mask3 & ...

3. **Majority**: A parameter is important if >50% of tasks use it
   - Middle ground between union and intersection
   - result = (sum(masks) >= threshold * num_tasks)

=== SIGNAL VS NOISE ===

- **Signal (masked)**: Parameters where mask is True - task-specific knowledge
- **Noise (unmasked)**: Parameters where mask is False - less important

SVD-Hybrid can optionally process both regions separately.

=== EXAMPLE ===

    >>> from mask_loader import load_task_masks, combine_masks
    >>> 
    >>> # Load masks for all tasks
    >>> task_masks = load_task_masks("./masks", ["Cars", "DTD", "EuroSAT"])
    >>> 
    >>> # Combine using union strategy
    >>> combined = combine_masks(task_masks, strategy="union")
    >>> 
    >>> # Apply to a parameter
    >>> signal = apply_mask_to_tensor(delta, combined["layer1.weight"])
"""

import torch
import numpy as np
import os
from typing import Dict, List, Optional, Union
from pathlib import Path


def state_dict_to_vector(
    state_dict: Dict[str, torch.Tensor],
    remove_keys: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Flatten a state dict into a single 1D vector.
    
    This is a utility function for converting model state dicts or mask dicts
    into flat vectors, which is useful for storage and comparison operations.
    
    Args:
        state_dict: Dictionary mapping parameter names to tensors
        remove_keys: Optional list of keys to skip (e.g., non-learnable buffers)
        
    Returns:
        1D tensor containing all flattened parameters concatenated
        
    Example:
        >>> state_dict = {"layer1.weight": torch.randn(10, 5), "layer1.bias": torch.randn(10)}
        >>> vector = state_dict_to_vector(state_dict)
        >>> vector.shape  # torch.Size([60])  # 10*5 + 10
    """
    if remove_keys is None:
        remove_keys = []
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(state_dict.keys())
    
    # Flatten and concatenate all tensors
    flat_tensors = []
    for key in sorted_keys:
        if key in remove_keys:
            continue
        tensor = state_dict[key]
        flat_tensors.append(tensor.flatten())
    
    if not flat_tensors:
        return torch.tensor([])
    
    return torch.cat(flat_tensors)


def vector_to_state_dict(
    vector: torch.Tensor,
    reference_state_dict: Dict[str, torch.Tensor],
    remove_keys: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct a state dict from a flat vector using a reference for shapes.
    
    This is the inverse of state_dict_to_vector. It takes a flat vector and
    reconstructs a state dict with the same structure as the reference.
    
    This is used to convert flat mask vectors from the original TALL mask format
    back into parameter-keyed dictionaries.
    
    Args:
        vector: 1D tensor containing flattened parameters
        reference_state_dict: A state dict with the target structure and shapes
        remove_keys: Keys that were removed during flattening (should match)
        
    Returns:
        Dictionary mapping parameter names to reshaped tensors
        
    Example:
        >>> reference = {"layer1.weight": torch.randn(10, 5), "layer1.bias": torch.randn(10)}
        >>> vector = torch.randn(60)  # 10*5 + 10
        >>> reconstructed = vector_to_state_dict(vector, reference)
        >>> reconstructed["layer1.weight"].shape  # torch.Size([10, 5])
    """
    if remove_keys is None:
        remove_keys = []
    
    # Sort keys for consistent ordering (same as state_dict_to_vector)
    sorted_keys = sorted(reference_state_dict.keys())
    
    result = {}
    offset = 0
    
    for key in sorted_keys:
        if key in remove_keys:
            continue
        
        ref_tensor = reference_state_dict[key]
        num_elements = ref_tensor.numel()
        
        # Extract slice from vector
        slice_values = vector[offset:offset + num_elements]
        
        # Reshape to match reference shape
        result[key] = slice_values.reshape(ref_tensor.shape)
        
        offset += num_elements
    
    return result


def load_tall_mask_file(
    mask_path: str,
    reference_state_dict: Dict[str, torch.Tensor],
    remove_keys: Optional[List[str]] = None,
    device: str = "cpu"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load TALL masks from the original format (single .npy file with packed bits).
    
    The original TALL mask format stores all task masks in a single file:
    - File format: .npy file loaded with torch.load() 
    - Structure: Dictionary with task names as keys
    - Values: Packed bit arrays (via np.packbits) for each task's mask
    
    The 8 standard tasks are: Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
    
    Args:
        mask_path: Path to the TALL mask file (e.g., "TALL_mask_8task.npy")
        reference_state_dict: A model state dict used to determine parameter shapes
                             for converting flat vectors back to state dicts
        remove_keys: Keys to exclude when reconstructing state dicts
        device: Device to load masks to
        
    Returns:
        Dictionary mapping task_name -> parameter_name -> boolean mask tensor
        
    Example:
        >>> base_state = torch.load("base_model.pt")
        >>> masks = load_tall_mask_file("TALL_mask_8task.npy", base_state)
        >>> masks["Cars"]["layer1.weight"].shape
        torch.Size([512, 1024])
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"TALL mask file not found: {mask_path}")
    
    if remove_keys is None:
        remove_keys = []
    
    # Load the packed mask dictionary
    # The original format uses torch.load even for .npy files
    packed_masks = torch.load(mask_path, map_location=device, weights_only=False)
    
    # Convert from packed format to state dicts
    task_masks = {}
    
    for task_name, packed_mask in packed_masks.items():
        # Unpack the bits: np.packbits was used to compress the boolean mask
        # np.unpackbits converts back to uint8 array of 0s and 1s
        if isinstance(packed_mask, np.ndarray):
            unpacked = np.unpackbits(packed_mask)
        else:
            # Handle case where it's already a torch tensor
            unpacked = np.unpackbits(packed_mask.numpy())
        
        # Convert to torch tensor
        mask_vector = torch.from_numpy(unpacked).to(device)
        
        # Calculate expected length from reference state dict
        expected_length = sum(
            v.numel() for k, v in reference_state_dict.items() 
            if k not in remove_keys
        )
        
        # Trim to expected length (packbits pads to multiple of 8)
        mask_vector = mask_vector[:expected_length]
        
        # Convert flat vector to state dict format
        mask_state_dict = vector_to_state_dict(
            mask_vector, 
            reference_state_dict, 
            remove_keys=remove_keys
        )
        
        # Ensure boolean dtype
        mask_state_dict = {k: v.bool() for k, v in mask_state_dict.items()}
        
        task_masks[task_name] = mask_state_dict
    
    return task_masks


def load_single_mask(mask_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load a single tall_masks binary mask file.
    
    Mask files are PyTorch state dicts where each key is a parameter name
    and each value is a boolean tensor with the same shape as that parameter.
    
    Args:
        mask_path: Path to mask file (.pt or .pth)
        device: Device to load mask to ("cpu" or "cuda")
        
    Returns:
        Dictionary mapping parameter names to boolean mask tensors
        
    Example:
        >>> mask = load_single_mask("./masks/Cars_mask.pt")
        >>> mask["layer1.weight"].dtype
        torch.bool
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load the mask dictionary
    mask_dict = torch.load(mask_path, map_location=device, weights_only=False)
    
    # Ensure all masks are boolean (convert if needed)
    for key, mask in mask_dict.items():
        if mask.dtype != torch.bool:
            mask_dict[key] = mask.bool()
    
    return mask_dict


def load_task_masks(
    mask_dir: str,
    task_names: List[str],
    device: str = "cpu",
    reference_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    remove_keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load masks for multiple tasks.
    
    Supports two mask formats:
    
    1. **Original TALL mask format** (preferred when reference_state_dict provided):
       - Single file containing all task masks: TALL_mask_{n}task.npy
       - Masks are packed bits that need unpacking with np.unpackbits
       - Requires reference_state_dict to reconstruct parameter shapes
    
    2. **Individual mask files**:
       - {task_name}_mask.pt (e.g., "Cars_mask.pt")
       - {task_name}.pt (e.g., "Cars.pt")
       - {task_name}/mask.pt (e.g., "Cars/mask.pt")
    
    If no mask is found for a task, a warning is printed and None is stored.
    
    Args:
        mask_dir: Directory containing mask files
        task_names: List of task identifiers to load
        device: Device to load masks to
        reference_state_dict: A model state dict for shape reference when loading
                             the original TALL mask format. If None, only individual
                             mask files will be searched.
        remove_keys: Keys to exclude when loading TALL format masks
        
    Returns:
        Dictionary mapping task_name -> parameter_name -> boolean mask
        Tasks without masks have None as their value
        
    Example:
        >>> # Load with individual mask files
        >>> masks = load_task_masks("./masks", ["Cars", "DTD", "EuroSAT"])
        >>> masks["Cars"]["layer1.weight"].shape
        torch.Size([512, 1024])
        
        >>> # Load from original TALL mask format
        >>> base_state = torch.load("base_model.pt")
        >>> masks = load_task_masks("./masks", ["Cars", "DTD"], reference_state_dict=base_state)
    """
    task_masks = {}
    num_tasks = len(task_names)
    
    # First, try to find the combined TALL mask file
    tall_mask_paths = [
        os.path.join(mask_dir, f"TALL_mask_{num_tasks}task.npy"),
        os.path.join(mask_dir, f"TALL_mask_{num_tasks}tasks.npy"),
        os.path.join(mask_dir, f"tall_mask_{num_tasks}task.npy"),
        os.path.join(mask_dir, f"tall_mask_{num_tasks}tasks.npy"),
    ]
    
    tall_mask_file = None
    for path in tall_mask_paths:
        if os.path.exists(path):
            tall_mask_file = path
            break
    
    if tall_mask_file is not None and reference_state_dict is not None:
        # Load using original TALL mask format
        print(f"Loading TALL masks from: {tall_mask_file}")
        all_task_masks = load_tall_mask_file(
            tall_mask_file,
            reference_state_dict,
            remove_keys=remove_keys,
            device=device
        )
        
        # Extract only the requested tasks
        for task_name in task_names:
            if task_name in all_task_masks:
                task_masks[task_name] = all_task_masks[task_name]
            else:
                print(f"Warning: Task {task_name} not found in TALL mask file")
                task_masks[task_name] = None
        
        return task_masks
    
    # Fall back to individual mask files
    for task_name in task_names:
        # Try various naming conventions
        possible_paths = [
            os.path.join(mask_dir, f"{task_name}_mask.pt"),
            os.path.join(mask_dir, f"{task_name}.pt"),
            os.path.join(mask_dir, task_name, "mask.pt"),
        ]
        
        # Find first existing path
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
    """
    Compute union of binary masks (OR operation).
    
    A parameter is included if it's important for ANY task.
    
    Args:
        masks: List of boolean mask tensors (same shape)
        
    Returns:
        Union mask (True where any input is True)
    """
    if not masks:
        raise ValueError("Empty mask list")
    
    result = masks[0].clone()
    for mask in masks[1:]:
        result = result | mask  # OR operation
    
    return result


def compute_intersection_mask(masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute intersection of binary masks (AND operation).
    
    A parameter is included only if it's important for ALL tasks.
    
    Args:
        masks: List of boolean mask tensors (same shape)
        
    Returns:
        Intersection mask (True where all inputs are True)
    """
    if not masks:
        raise ValueError("Empty mask list")
    
    result = masks[0].clone()
    for mask in masks[1:]:
        result = result & mask  # AND operation
    
    return result


def compute_majority_mask(masks: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
    """
    Compute majority voting mask.
    
    A parameter is included if it's important for at least threshold fraction of tasks.
    
    Args:
        masks: List of boolean mask tensors
        threshold: Fraction of masks that must be True (default: 0.5 = majority)
        
    Returns:
        Majority mask (True where >= threshold fraction are True)
        
    Example:
        >>> # With 4 tasks and threshold=0.5:
        >>> # Parameter needs 2+ tasks to mark it as important
    """
    if not masks:
        raise ValueError("Empty mask list")
    
    # Stack masks and convert to float for summation
    stacked = torch.stack([m.float() for m in masks], dim=0)
    
    # Count how many masks have True for each position
    vote_sum = stacked.sum(dim=0)
    
    # Apply threshold: need at least (threshold * num_masks) votes
    result = vote_sum >= (threshold * len(masks))
    
    return result


def combine_masks(
    task_masks: Dict[str, Dict[str, torch.Tensor]],
    strategy: str = "union",
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Combine masks from multiple tasks using specified strategy.
    
    For each parameter, collects masks from all tasks that have that parameter,
    then combines them using the specified strategy.
    
    Args:
        task_masks: Dictionary mapping task_name -> parameter_name -> mask
        strategy: Combination strategy - "union", "intersection", or "majority"
        device: Device for computation
        
    Returns:
        Dictionary mapping parameter_name -> combined boolean mask
        
    Example:
        >>> task_masks = {
        ...     "Cars": {"layer1.weight": mask1},
        ...     "DTD": {"layer1.weight": mask2}
        ... }
        >>> combined = combine_masks(task_masks, strategy="union")
        >>> # combined["layer1.weight"] = mask1 | mask2
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
                # Task has no mask file, skip
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
    
    Extracts values from tensor where mask is True, returning them as a 1D tensor.
    This is used to get the "signal" (important) portion of a parameter.
    
    Args:
        tensor: Input tensor (any shape)
        mask: Binary mask (same shape as tensor, True = keep)
        
    Returns:
        1D tensor containing values where mask is True
        
    Example:
        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = torch.tensor([[True, False, True], [False, True, False]])
        >>> signal = apply_mask_to_tensor(tensor, mask)
        >>> signal  # tensor([1, 3, 5])
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tensor {tensor.shape} vs mask {mask.shape}")
    
    # Flatten both for indexing
    flat_tensor = tensor.flatten()
    flat_mask = mask.flatten()
    
    # Return values where mask is True
    return flat_tensor[flat_mask]


def get_unmasked_portion(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Get unmasked (noise) portion of tensor.
    
    Extracts values from tensor where mask is False, returning them as a 1D tensor.
    This is used to get the "noise" (less important) portion of a parameter.
    
    Args:
        tensor: Input tensor (any shape)
        mask: Binary mask (same shape as tensor)
        
    Returns:
        1D tensor containing values where mask is False
        
    Example:
        >>> tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> mask = torch.tensor([[True, False, True], [False, True, False]])
        >>> noise = get_unmasked_portion(tensor, mask)
        >>> noise  # tensor([2, 4, 6])
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"Shape mismatch: tensor {tensor.shape} vs mask {mask.shape}")
    
    flat_tensor = tensor.flatten()
    flat_mask = mask.flatten()
    
    # Return values where mask is False (using ~mask for negation)
    return flat_tensor[~flat_mask]


def reconstruct_from_masked(
    masked_values: torch.Tensor,
    unmasked_values: Optional[torch.Tensor],
    mask: torch.Tensor,
    original_shape: torch.Size
) -> torch.Tensor:
    """
    Reconstruct full tensor from masked and optionally unmasked portions.
    
    This is the inverse of apply_mask_to_tensor + get_unmasked_portion.
    Places values back into their original positions based on the mask.
    
    === HOW IT WORKS ===
    
    1. Create a flat result tensor filled with zeros
    2. Put masked_values where mask is True
    3. Put unmasked_values where mask is False (if provided)
    4. Reshape to original shape
    
    Args:
        masked_values: Signal values (1D tensor, length = mask.sum())
        unmasked_values: Noise values (1D tensor, length = (~mask).sum()), optional
        mask: Binary mask with original shape
        original_shape: Target shape for output tensor
        
    Returns:
        Reconstructed tensor with original shape
        
    Example:
        >>> # Separate signal and noise
        >>> signal = apply_mask_to_tensor(tensor, mask)
        >>> noise = get_unmasked_portion(tensor, mask)
        >>> 
        >>> # Later, reconstruct
        >>> reconstructed = reconstruct_from_masked(signal, noise, mask, tensor.shape)
        >>> torch.allclose(reconstructed, tensor)
        True
    """
    flat_mask = mask.flatten()
    
    # Create result tensor filled with zeros
    result_flat = torch.zeros_like(flat_mask, dtype=masked_values.dtype, device=masked_values.device)
    
    # Fill in masked (signal) portion
    result_flat[flat_mask] = masked_values
    
    # Fill in unmasked (noise) portion if provided
    if unmasked_values is not None:
        result_flat[~flat_mask] = unmasked_values
    
    # Reshape to original shape
    return result_flat.view(original_shape)
