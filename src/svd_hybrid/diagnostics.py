"""
Diagnostics: reconstruction error, energy metrics, compression ratios.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .rtvq import RTVQQuantizer, estimate_compression_ratio


def compute_reconstruction_error(
    original_delta: torch.Tensor,
    reconstructed_delta: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.
    
    Args:
        original_delta: Original delta vector
        reconstructed_delta: Reconstructed delta vector
        
    Returns:
        Dictionary of error metrics
    """
    error = original_delta - reconstructed_delta
    
    original_norm = original_delta.norm().item()
    error_norm = error.norm().item()
    
    relative_error = error_norm / original_norm if original_norm > 1e-10 else 0
    
    return {
        "absolute_error": error_norm,
        "relative_error": relative_error,
        "max_absolute_error": error.abs().max().item(),
        "mean_absolute_error": error.abs().mean().item(),
        "original_norm": original_norm,
        "reconstructed_norm": reconstructed_delta.norm().item()
    }


def compute_parameter_diagnostics(
    param_name: str,
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    compressed_params: Dict[str, Dict],
    basis: Dict,
    mask: Optional[torch.Tensor],
    quantizer: RTVQQuantizer,
    device: str = "cpu"
) -> Dict:
    """
    Compute diagnostics for a single parameter.
    
    Args:
        param_name: Parameter name
        task_vectors: Original task vectors
        compressed_params: Compressed coefficients
        basis: Basis for this parameter
        mask: Binary mask (optional)
        quantizer: RTVQ quantizer
        device: Device for computation
        
    Returns:
        Diagnostics dictionary
    """
    from .mask_loader import apply_mask_to_tensor
    from .compress import project_to_basis
    
    diagnostics = {
        "param_name": param_name,
        "original_shape": None,
        "masked_size": 0,
        "unmasked_size": 0,
        "reconstruction_errors": {},
        "compression_ratios": {}
    }
    
    # Get basis info
    basis_masked = basis.get("masked")
    if basis_masked is None:
        return diagnostics
    
    # Get original shape from first task
    first_task = next(iter(task_vectors.keys()))
    if param_name in task_vectors[first_task]:
        original_shape = task_vectors[first_task][param_name].shape
        diagnostics["original_shape"] = list(original_shape)
    
    # Compute masked size
    if mask is not None:
        diagnostics["masked_size"] = int(mask.sum().item())
        diagnostics["unmasked_size"] = int((~mask).sum().item())
    else:
        diagnostics["masked_size"] = np.prod(diagnostics["original_shape"])
    
    # Basis statistics
    diagnostics["basis"] = {
        "k": basis_masked["k"],
        "D": basis_masked["D"],
        "N": basis_masked["N"],
        "energy_retained": basis_masked["energy_retained"]
    }
    
    # Per-task reconstruction errors
    task_errors = []
    
    for task_name in task_vectors.keys():
        if param_name not in task_vectors[task_name]:
            continue
        
        if task_name not in compressed_params:
            continue
        
        # Get original masked delta
        original_delta = task_vectors[task_name][param_name]
        
        if mask is not None and mask.shape == original_delta.shape:
            original_masked = apply_mask_to_tensor(original_delta, mask)
        else:
            original_masked = original_delta.flatten()
        
        # Reconstruct from compressed
        artifact = compressed_params[task_name]
        if artifact.get("masked") is None:
            continue
        
        c_high = artifact["masked"]["c_high_fp16"].to(device).float()
        c_low_quant = artifact["masked"]["c_low_quant"]
        c_low = quantizer.dequantize(c_low_quant, device=device).float()
        
        # Reconstruct
        U_high = basis_masked["U_high"].to(device).float()
        U_low = basis_masked["U_low"].to(device).float()
        reconstructed = U_high @ c_high + U_low @ c_low
        
        # Compute error
        error_metrics = compute_reconstruction_error(original_masked, reconstructed)
        task_errors.append(error_metrics["relative_error"])
        
        diagnostics["reconstruction_errors"][task_name] = error_metrics
        
        # Compute compression ratio
        compression_ratio = estimate_compression_ratio(c_low, c_low_quant)
        diagnostics["compression_ratios"][task_name] = compression_ratio
    
    # Summary statistics
    if task_errors:
        diagnostics["mean_relative_error"] = float(np.mean(task_errors))
        diagnostics["std_relative_error"] = float(np.std(task_errors))
        diagnostics["max_relative_error"] = float(np.max(task_errors))
        diagnostics["min_relative_error"] = float(np.min(task_errors))
    
    return diagnostics


def compute_all_diagnostics(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    compressed_all: Dict[str, Dict[str, Dict]],
    bases: Dict[str, Dict],
    masks: Dict[str, torch.Tensor],
    config,
    device: str = "cpu"
) -> Dict:
    """
    Compute diagnostics for all parameters.
    
    Args:
        task_vectors: Original task vectors
        compressed_all: All compressed coefficients
        bases: All bases
        masks: All masks
        config: SVDHybridConfig
        device: Device for computation
        
    Returns:
        Complete diagnostics dictionary
    """
    quantizer = RTVQQuantizer(
        num_bits=config.svd_low_bits,
        num_stages=config.svd_rtvq_stages
    )
    
    diagnostics = {
        "config": {
            "svd_energy_threshold": config.svd_energy_threshold,
            "svd_max_rank": config.svd_max_rank,
            "svd_low_bits": config.svd_low_bits,
            "svd_rtvq_stages": config.svd_rtvq_stages,
            "svd_mask_strategy": config.svd_mask_strategy,
            "svd_weighting": config.svd_weighting
        },
        "per_parameter": {},
        "summary": {}
    }
    
    # Compute per-parameter diagnostics
    param_names = sorted(bases.keys())
    
    for param_name in param_names:
        if param_name not in compressed_all:
            continue
        
        param_diag = compute_parameter_diagnostics(
            param_name,
            task_vectors,
            compressed_all[param_name],
            bases[param_name],
            masks.get(param_name),
            quantizer,
            device
        )
        
        diagnostics["per_parameter"][param_name] = param_diag
    
    # Compute summary statistics
    all_ranks = []
    all_energy_retained = []
    all_mean_errors = []
    all_compression_ratios = []
    
    for param_diag in diagnostics["per_parameter"].values():
        if "basis" in param_diag:
            all_ranks.append(param_diag["basis"]["k"])
            all_energy_retained.append(param_diag["basis"]["energy_retained"])
        
        if "mean_relative_error" in param_diag:
            all_mean_errors.append(param_diag["mean_relative_error"])
        
        # Average compression ratio for this parameter
        if param_diag.get("compression_ratios"):
            param_avg_ratio = np.mean(list(param_diag["compression_ratios"].values()))
            all_compression_ratios.append(param_avg_ratio)
    
    diagnostics["summary"] = {
        "num_parameters": len(diagnostics["per_parameter"]),
        "average_rank": float(np.mean(all_ranks)) if all_ranks else 0,
        "std_rank": float(np.std(all_ranks)) if all_ranks else 0,
        "average_energy_retained": float(np.mean(all_energy_retained)) if all_energy_retained else 0,
        "average_reconstruction_error": float(np.mean(all_mean_errors)) if all_mean_errors else 0,
        "average_compression_ratio": float(np.mean(all_compression_ratios)) if all_compression_ratios else 0
    }
    
    return diagnostics


def compute_coefficient_histograms(
    compressed_params: Dict[str, Dict],
    quantizer: RTVQQuantizer,
    num_bins: int = 50,
    device: str = "cpu"
) -> Dict:
    """
    Compute histograms of coefficient magnitudes.
    
    Args:
        compressed_params: Compressed coefficients for one parameter
        quantizer: RTVQ quantizer
        num_bins: Number of histogram bins
        device: Device for computation
        
    Returns:
        Histogram data
    """
    all_c_high = []
    all_c_low = []
    
    for task_name, artifact in compressed_params.items():
        if artifact.get("masked") is None:
            continue
        
        c_high = artifact["masked"]["c_high_fp16"].to(device).float()
        all_c_high.append(c_high.flatten())
        
        c_low_quant = artifact["masked"]["c_low_quant"]
        c_low = quantizer.dequantize(c_low_quant, device=device).float()
        all_c_low.append(c_low.flatten())
    
    if not all_c_high:
        return {}
    
    # Concatenate all coefficients
    c_high_cat = torch.cat(all_c_high, dim=0).cpu().numpy()
    c_low_cat = torch.cat(all_c_low, dim=0).cpu().numpy()
    
    # Compute histograms
    c_high_hist, c_high_bins = np.histogram(np.abs(c_high_cat), bins=num_bins)
    c_low_hist, c_low_bins = np.histogram(np.abs(c_low_cat), bins=num_bins)
    
    return {
        "c_high": {
            "counts": c_high_hist.tolist(),
            "bin_edges": c_high_bins.tolist(),
            "mean": float(np.mean(np.abs(c_high_cat))),
            "std": float(np.std(np.abs(c_high_cat))),
            "max": float(np.max(np.abs(c_high_cat)))
        },
        "c_low": {
            "counts": c_low_hist.tolist(),
            "bin_edges": c_low_bins.tolist(),
            "mean": float(np.mean(np.abs(c_low_cat))),
            "std": float(np.std(np.abs(c_low_cat))),
            "max": float(np.max(np.abs(c_low_cat)))
        }
    }


def print_diagnostics_summary(diagnostics: Dict):
    """
    Print human-readable summary of diagnostics.
    
    Args:
        diagnostics: Diagnostics dictionary
    """
    print("\n" + "="*60)
    print("SVD-Hybrid Diagnostics Summary")
    print("="*60)
    
    summary = diagnostics.get("summary", {})
    
    print(f"\nNumber of parameters: {summary.get('num_parameters', 0)}")
    print(f"Average rank: {summary.get('average_rank', 0):.2f} ± {summary.get('std_rank', 0):.2f}")
    print(f"Average energy retained: {summary.get('average_energy_retained', 0):.4f}")
    print(f"Average reconstruction error: {summary.get('average_reconstruction_error', 0):.6f}")
    print(f"Average compression ratio: {summary.get('average_compression_ratio', 0):.2f}x")
    
    print("\n" + "="*60)
