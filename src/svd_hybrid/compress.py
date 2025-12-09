import torch
from typing import Dict, List, Tuple, Optional
from .rtvq import RTVQQuantizer


def project_to_basis(
    delta: torch.Tensor,
    U_high: torch.Tensor,
    U_low: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert to float32 for computation precision
    delta_f = delta.float()
    U_high_f = U_high.float()
    U_low_f = U_low.float()
    
    # Project: c = U^T × delta
    c_high = U_high_f.T @ delta_f  # [k]
    c_low = U_low_f.T @ delta_f    # [D-k]
    
    return c_high, c_low


def compress_single_task(
    task_delta: torch.Tensor,
    U_high: torch.Tensor,
    U_low: torch.Tensor,
    quantizer: RTVQQuantizer,
    device: str = "cpu",
    mean: Optional[torch.Tensor] = None
) -> Dict:
   
    # Step 0: Subtract mean if provided (for centered SVD basis)
    delta_to_project = task_delta
    if mean is not None:
        mean_vec = mean.squeeze()  # Handle both [D] and [D x 1] shapes
        delta_to_project = task_delta - mean_vec
    
    # Step 1: Project (centered) delta to coefficient space
    c_high, c_low = project_to_basis(delta_to_project, U_high, U_low)
    
    # Step 2: Store c_high in FP16 (half precision)
    # This preserves accuracy for important coefficients while halving memory
    if c_high.dtype != torch.float16:
        c_high_fp16 = c_high.half()
    else:
        c_high_fp16 = c_high
    
    # Step 3: Quantize c_low with RTVQ
    # Move to CPU for storage efficiency
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
   
    compressed_tasks = {}
    
    # Extract mean vectors if present (for centered SVD basis)
    mean_masked = basis_masked.get("mean") if basis_masked is not None else None
    mean_unmasked = basis_unmasked.get("mean") if basis_unmasked is not None else None
    
    for task_name in task_deltas_masked.keys():
        artifact = {}
        
        # Compress masked region
        if basis_masked is not None and len(task_deltas_masked[task_name]) > 0:
            masked_artifact = compress_single_task(
                task_deltas_masked[task_name],
                basis_masked["U_high"],
                basis_masked["U_low"],
                quantizer,
                device,
                mean=mean_masked
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
                device,
                mean=mean_unmasked
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
