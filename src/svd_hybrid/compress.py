"""
Compression pipeline: project deltas, quantize low-energy coefficients, store artifacts.
"""
import torch
from typing import Dict, List, Tuple, Optional
from .rtvq import RTVQQuantizer


def project_to_basis(
    delta: torch.Tensor,
    U_high: torch.Tensor,
    U_low: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project delta vector onto basis.
    
    Args:
        delta: Delta vector [D]
        U_high: High-energy basis [D x k]
        U_low: Low-energy basis [D x (D-k)]
        
    Returns:
        Tuple of (c_high, c_low) coefficients
    """
    delta_f = delta.float()
    U_high_f = U_high.float()
    U_low_f = U_low.float()
    
    c_high = U_high_f.T @ delta_f  # [k]
    c_low = U_low_f.T @ delta_f    # [D-k]
    
    return c_high, c_low


def compress_single_task(
    task_delta: torch.Tensor,
    U_high: torch.Tensor,
    U_low: torch.Tensor,
    quantizer: RTVQQuantizer,
    device: str = "cpu"
) -> Dict:
    """
    Compress a single task's delta vector.
    
    Args:
        task_delta: Task delta vector [D]
        U_high: High-energy basis [D x k]
        U_low: Low-energy basis [D x (D-k)]
        quantizer: RTVQ quantizer
        device: Device for computation
        
    Returns:
        Compression artifact dictionary
    """
    # Project
    c_high, c_low = project_to_basis(task_delta, U_high, U_low)
    
    # Store c_high in FP16
    if c_high.dtype != torch.float16:
        c_high_fp16 = c_high.half()
    else:
        c_high_fp16 = c_high
    
    # Quantize c_low
    c_low_quant = quantizer.quantize(c_low.cpu())
    
    return {
        "c_high_fp16": c_high_fp16.cpu(),
        "c_low_quant": c_low_quant
    }


def compress_masked_regions(
    task_deltas_masked: Dict[str, torch.Tensor],
    task_deltas_unmasked: Optional[Dict[str, torch.Tensor]],
    basis_masked: Dict,
    basis_unmasked: Optional[Dict],
    quantizer: RTVQQuantizer,
    device: str = "cpu"
) -> Dict[str, Dict]:
    """
    Compress masked and optionally unmasked regions for all tasks.
    
    Args:
        task_deltas_masked: Dictionary mapping task_name -> masked delta
        task_deltas_unmasked: Dictionary mapping task_name -> unmasked delta (optional)
        basis_masked: Masked basis dictionary
        basis_unmasked: Unmasked basis dictionary (optional)
        quantizer: RTVQ quantizer
        device: Device for computation
        
    Returns:
        Dictionary mapping task_name -> compression artifacts
    """
    compressed_tasks = {}
    
    for task_name in task_deltas_masked.keys():
        artifact = {}
        
        # Compress masked region
        if basis_masked is not None and len(task_deltas_masked[task_name]) > 0:
            masked_artifact = compress_single_task(
                task_deltas_masked[task_name],
                basis_masked["U_high"],
                basis_masked["U_low"],
                quantizer,
                device
            )
            artifact["masked"] = masked_artifact
        else:
            artifact["masked"] = None
        
        # Compress unmasked region if available
        if (basis_unmasked is not None and 
            task_deltas_unmasked is not None and 
            task_name in task_deltas_unmasked and
            len(task_deltas_unmasked[task_name]) > 0):
            
            unmasked_artifact = compress_single_task(
                task_deltas_unmasked[task_name],
                basis_unmasked["U_high"],
                basis_unmasked["U_low"],
                quantizer,
                device
            )
            artifact["unmasked"] = unmasked_artifact
        else:
            artifact["unmasked"] = None
        
        compressed_tasks[task_name] = artifact
    
    return compressed_tasks


def compress_parameter(
    param_name: str,
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    mask: Optional[torch.Tensor],
    basis: Dict,
    quantizer: RTVQQuantizer,
    include_noise: bool = False,
    min_mask_size: int = 10,
    device: str = "cpu"
) -> Dict:
    """
    Compress a single parameter across all tasks.
    
    Args:
        param_name: Parameter name
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta
        mask: Binary mask for this parameter (optional)
        basis: Basis dictionary (contains "masked" and optionally "noise")
        quantizer: RTVQ quantizer
        include_noise: Whether to process noise region
        min_mask_size: Minimum mask size to process
        device: Device for computation
        
    Returns:
        Compression artifacts for this parameter
    """
    from .mask_loader import apply_mask_to_tensor, get_unmasked_portion
    
    # Extract deltas for this parameter
    task_deltas = {}
    for task_name, task_vector in task_vectors.items():
        if param_name in task_vector:
            task_deltas[task_name] = task_vector[param_name]
    
    if not task_deltas:
        return None
    
    # Apply mask
    task_deltas_masked = {}
    task_deltas_unmasked = {}
    
    for task_name, delta in task_deltas.items():
        if mask is not None and mask.shape == delta.shape:
            # Extract masked portion
            if mask.sum() >= min_mask_size:
                masked = apply_mask_to_tensor(delta, mask)
                task_deltas_masked[task_name] = masked
            else:
                task_deltas_masked[task_name] = torch.tensor([])
            
            # Extract unmasked portion if needed
            if include_noise:
                unmasked = get_unmasked_portion(delta, mask)
                task_deltas_unmasked[task_name] = unmasked
        else:
            # No mask, use entire delta
            task_deltas_masked[task_name] = delta.flatten()
    
    # Compress
    basis_masked = basis.get("masked")
    basis_unmasked = basis.get("noise") if include_noise else None
    
    compressed = compress_masked_regions(
        task_deltas_masked,
        task_deltas_unmasked if include_noise else None,
        basis_masked,
        basis_unmasked,
        quantizer,
        device
    )
    
    return compressed


def compress_all_parameters(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    masks: Dict[str, torch.Tensor],
    bases: Dict[str, Dict],
    config,
    device: str = "cpu"
) -> Dict[str, Dict]:
    """
    Compress all parameters across all tasks.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta
        masks: Dictionary mapping parameter_name -> mask
        bases: Dictionary mapping parameter_name -> basis
        config: SVDHybridConfig object
        device: Device for computation
        
    Returns:
        Dictionary mapping parameter_name -> task_name -> compression artifacts
    """
    quantizer = RTVQQuantizer(
        num_bits=config.svd_low_bits,
        num_stages=config.svd_rtvq_stages
    )
    
    compressed_params = {}
    
    param_names = sorted(bases.keys())
    
    for param_name in param_names:
        mask = masks.get(param_name)
        basis = bases[param_name]
        
        compressed = compress_parameter(
            param_name,
            task_vectors,
            mask,
            basis,
            quantizer,
            include_noise=config.svd_include_noise,
            min_mask_size=config.svd_min_mask_size,
            device=device
        )
        
        if compressed is not None:
            compressed_params[param_name] = compressed
    
    return compressed_params
