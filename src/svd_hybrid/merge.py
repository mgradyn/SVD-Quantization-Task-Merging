"""
Merging logic with weighted averaging and reconstruction.

=== TUTORIAL: The Merging Process ===

This module implements the merging step of SVD-Hybrid, which combines
compressed task representations into a single merged model.

=== THE MERGING PROCESS ===

1. **Dequantize**: Convert quantized low-energy coefficients back to floats
2. **Weighted Average**: Combine coefficients across tasks with weights
3. **Reconstruct**: Transform averaged coefficients back to parameter space
4. **Apply Deltas**: Add merged deltas to base model

=== WEIGHTED AVERAGING ===

For each coefficient position:
    c_avg = Σ (weight_i × c_i) for all tasks

Where weights sum to 1 (e.g., uniform: 1/N, performance-based: softmax)

=== RECONSTRUCTION FROM COEFFICIENTS ===

Merged delta is reconstructed from averaged coefficients:
    merged_delta = U_high × c_high_avg + U_low × c_low_avg

This reverses the projection done during compression.

=== FINAL MODEL ===

The merged model is:
    merged_model = base_model + merged_delta

=== CLUSTER-BASED MERGING ===

With cluster-based weighting, the process is hierarchical:
1. Merge within clusters (similar tasks averaged)
2. Merge cluster results (cluster centroids averaged)

This can improve results when tasks naturally group together.

=== EXAMPLE ===

    >>> from merge import merge_all_parameters, apply_merged_deltas
    >>> 
    >>> # Merge all parameters
    >>> merged_deltas = merge_all_parameters(
    ...     compressed, bases, masks, weights, shapes, config
    ... )
    >>> 
    >>> # Apply to base model
    >>> merged_model = apply_merged_deltas(base_state_dict, merged_deltas)
"""

import torch
from typing import Dict, List, Optional, Tuple
from .rtvq import RTVQQuantizer


def dequantize_and_average(
    compressed_coeffs: Dict[str, Dict],
    weights: Dict[str, float],
    quantizer: RTVQQuantizer,
    region: str = "masked",
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize low coefficients and compute weighted average of high and low.
    
    This function:
    1. Extracts coefficients from each task's compressed representation
    2. Dequantizes the low-energy coefficients (RTVQ → float)
    3. Computes weighted average across all tasks
    
    Args:
        compressed_coeffs: Dictionary mapping task_name -> compression artifacts
        weights: Dictionary mapping task_name -> weight (should sum to 1)
        quantizer: RTVQ quantizer for dequantization
        region: Which region to process - "masked" (signal) or "unmasked" (noise)
        device: Device for computation
        
    Returns:
        Tuple of:
            - avg_c_high: Weighted average of high-energy coefficients [k]
            - avg_c_low: Weighted average of low-energy coefficients [D-k]
        Or (None, None) if no valid coefficients found
    """
    task_names = sorted(compressed_coeffs.keys())
    
    # Collect coefficients from each task
    c_high_list = []
    c_low_list = []
    weight_list = []
    
    for task_name in task_names:
        artifact = compressed_coeffs[task_name]
        
        # Skip if this task has no data for this region
        if artifact is None or artifact.get(region) is None:
            continue
        
        region_artifact = artifact[region]
        
        # Get high-energy coefficients (already FP16, convert to FP32)
        c_high = region_artifact["c_high_fp16"].to(device).float()
        c_high_list.append(c_high)
        
        # Dequantize low-energy coefficients
        c_low_quant = region_artifact["c_low_quant"]
        c_low = quantizer.dequantize(c_low_quant, device=device).float()
        c_low_list.append(c_low)
        
        # Get this task's weight
        weight = weights.get(task_name, 1.0 / len(task_names))
        weight_list.append(weight)
    
    # Handle empty case
    if not c_high_list:
        return None, None
    
    # Normalize weights to sum to 1 (in case only subset of tasks are present)
    total_weight = sum(weight_list)
    weight_list = [w / total_weight for w in weight_list]
    
    # Compute weighted averages
    weight_tensor = torch.tensor(weight_list, device=device, dtype=torch.float32)
    
    # Stack coefficients: [N x k] and [N x (D-k)]
    c_high_stacked = torch.stack(c_high_list, dim=0)
    c_low_stacked = torch.stack(c_low_list, dim=0)
    
    # Reshape weights for broadcasting: [N x 1]
    weight_tensor_high = weight_tensor.view(-1, 1)
    weight_tensor_low = weight_tensor.view(-1, 1)
    
    # Weighted sum along task dimension
    avg_c_high = (c_high_stacked * weight_tensor_high).sum(dim=0)
    avg_c_low = (c_low_stacked * weight_tensor_low).sum(dim=0)
    
    return avg_c_high, avg_c_low


def reconstruct_from_coefficients(
    avg_c_high: torch.Tensor,
    avg_c_low: torch.Tensor,
    U_high: torch.Tensor,
    U_low: torch.Tensor,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Reconstruct delta from averaged coefficients.
    
    This is the inverse of the projection step. Transforms coefficients
    back to parameter space.
    
    === FORMULA ===
    
    delta = U_high × c_high + U_low × c_low
         = [D × k] × [k] + [D × (D-k)] × [D-k]
         = [D] + [D]
         = [D]
    
    Args:
        avg_c_high: Averaged high coefficients [k]
        avg_c_low: Averaged low coefficients [D-k]
        U_high: High-energy basis [D × k]
        U_low: Low-energy basis [D × (D-k)]
        device: Device for computation
        
    Returns:
        Reconstructed delta vector [D]
    """
    # Convert to float32 for computation
    U_high_f = U_high.to(device).float()
    U_low_f = U_low.to(device).float()
    
    # Reconstruct: delta = U × c
    part_high = U_high_f @ avg_c_high  # [D]
    part_low = U_low_f @ avg_c_low      # [D]
    
    return part_high + part_low


def merge_parameter(
    param_name: str,
    compressed_params: Dict[str, Dict],
    basis: Dict,
    weights: Dict[str, float],
    quantizer: RTVQQuantizer,
    original_shape: torch.Size,
    mask: Optional[torch.Tensor] = None,
    include_noise: bool = False,
    noise_shrink: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Merge a single parameter across tasks.
    
    Args:
        param_name: Parameter name
        compressed_params: Compressed coefficients for this parameter
        basis: Basis for this parameter
        weights: Task weights
        quantizer: RTVQ quantizer
        original_shape: Original parameter shape
        mask: Binary mask (optional)
        include_noise: Whether noise region was processed
        noise_shrink: Shrinkage factor for noise region
        device: Device for computation
        
    Returns:
        Merged delta tensor with original shape
    """
    from .mask_loader import reconstruct_from_masked
    
    # Merge masked region
    basis_masked = basis.get("masked")
    if basis_masked is not None:
        avg_c_high_masked, avg_c_low_masked = dequantize_and_average(
            compressed_params,
            weights,
            quantizer,
            region="masked",
            device=device
        )
        
        if avg_c_high_masked is not None:
            merged_masked = reconstruct_from_coefficients(
                avg_c_high_masked,
                avg_c_low_masked,
                basis_masked["U_high"],
                basis_masked["U_low"],
                device=device
            )
        else:
            merged_masked = None
    else:
        merged_masked = None
    
    # Merge noise region if included
    merged_unmasked = None
    if include_noise:
        basis_unmasked = basis.get("noise")
        if basis_unmasked is not None:
            avg_c_high_unmasked, avg_c_low_unmasked = dequantize_and_average(
                compressed_params,
                weights,
                quantizer,
                region="unmasked",
                device=device
            )
            
            if avg_c_high_unmasked is not None:
                merged_unmasked = reconstruct_from_coefficients(
                    avg_c_high_unmasked,
                    avg_c_low_unmasked,
                    basis_unmasked["U_high"],
                    basis_unmasked["U_low"],
                    device=device
                )
                
                # Apply noise shrinkage
                merged_unmasked = merged_unmasked * noise_shrink
    
    # Reconstruct full parameter
    if mask is not None and merged_masked is not None:
        merged_full = reconstruct_from_masked(
            merged_masked,
            merged_unmasked,
            mask,
            original_shape
        )
    elif merged_masked is not None:
        # No mask, just reshape
        merged_full = merged_masked.view(original_shape)
    else:
        # Fallback to zeros
        merged_full = torch.zeros(original_shape, device=device)
    
    return merged_full


def merge_all_parameters(
    compressed_all: Dict[str, Dict[str, Dict]],
    bases: Dict[str, Dict],
    masks: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    original_shapes: Dict[str, torch.Size],
    config,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Merge all parameters.
    
    Args:
        compressed_all: Dictionary mapping parameter_name -> task_name -> artifacts
        bases: Dictionary mapping parameter_name -> basis
        masks: Dictionary mapping parameter_name -> mask
        weights: Dictionary mapping task_name -> weight
        original_shapes: Dictionary mapping parameter_name -> shape
        config: SVDHybridConfig
        device: Device for computation
        
    Returns:
        Dictionary mapping parameter_name -> merged delta
    """
    quantizer = RTVQQuantizer(
        num_bits=config.svd_low_bits,
        num_stages=config.svd_rtvq_stages
    )
    
    merged_deltas = {}
    
    for param_name in sorted(compressed_all.keys()):
        compressed_params = compressed_all[param_name]
        basis = bases[param_name]
        mask = masks.get(param_name)
        original_shape = original_shapes[param_name]
        
        merged = merge_parameter(
            param_name,
            compressed_params,
            basis,
            weights,
            quantizer,
            original_shape,
            mask=mask,
            include_noise=config.svd_include_noise,
            noise_shrink=config.svd_noise_shrink,
            device=device
        )
        
        merged_deltas[param_name] = merged
    
    return merged_deltas


def apply_merged_deltas(
    base_state_dict: Dict[str, torch.Tensor],
    merged_deltas: Dict[str, torch.Tensor],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Apply merged deltas to base model to create merged model.
    
    The final step of model merging - adds the merged task vector to the
    base model to produce the multi-task merged model.
    
    === FORMULA ===
    
    merged_model[param] = base_model[param] + merged_delta[param]
    
    Parameters not in merged_deltas are copied unchanged from base model.
    
    Args:
        base_state_dict: Base model state dict
        merged_deltas: Merged delta vectors from merge_all_parameters
        device: Device for computation
        
    Returns:
        Merged model state dict ready to load into a model
        
    Example:
        >>> merged_state = apply_merged_deltas(base_state, merged_deltas)
        >>> model.load_state_dict(merged_state)
    """
    merged_state_dict = {}
    
    for param_name, base_param in base_state_dict.items():
        if param_name in merged_deltas:
            # Apply delta: merged = base + delta
            delta = merged_deltas[param_name].to(base_param.device)
            merged_param = base_param + delta
            merged_state_dict[param_name] = merged_param
        else:
            # Keep base value unchanged
            merged_state_dict[param_name] = base_param.clone()
    
    return merged_state_dict


def merge_with_clustering(
    compressed_all: Dict[str, Dict[str, Dict]],
    bases: Dict[str, Dict],
    masks: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    cluster_assignments: Dict[str, int],
    original_shapes: Dict[str, torch.Size],
    config,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Merge using cluster-based approach.
    
    First merges within clusters, then merges cluster results.
    
    Args:
        compressed_all: Compressed coefficients
        bases: Bases for all parameters
        masks: Masks for all parameters
        weights: Task weights
        cluster_assignments: Task cluster assignments
        original_shapes: Original parameter shapes
        config: SVDHybridConfig
        device: Device for computation
        
    Returns:
        Merged deltas
    """
    from .clustering import get_cluster_members
    
    # Get cluster members
    clusters = get_cluster_members(cluster_assignments)
    
    # Merge within each cluster
    cluster_merged = {}
    for cluster_id, member_names in clusters.items():
        # Filter weights for this cluster
        cluster_weights = {name: weights.get(name, 1.0) for name in member_names}
        total = sum(cluster_weights.values())
        cluster_weights = {k: v / total for k, v in cluster_weights.items()}
        
        # Filter compressed coefficients for this cluster
        cluster_compressed = {}
        for param_name, param_compressed in compressed_all.items():
            cluster_compressed[param_name] = {
                name: param_compressed[name]
                for name in member_names
                if name in param_compressed
            }
        
        # Merge this cluster
        cluster_merged[cluster_id] = merge_all_parameters(
            cluster_compressed,
            bases,
            masks,
            cluster_weights,
            original_shapes,
            config,
            device
        )
    
    # Compute cluster performance (average of member weights)
    cluster_performance = {}
    for cluster_id, member_names in clusters.items():
        avg_weight = sum(weights.get(name, 1.0) for name in member_names) / len(member_names)
        cluster_performance[cluster_id] = avg_weight
    
    # Merge across clusters
    from .clustering import merge_cluster_results
    final_merged = merge_cluster_results(cluster_merged, cluster_performance, device)
    
    return final_merged
