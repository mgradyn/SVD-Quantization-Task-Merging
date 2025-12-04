"""
Diagnostics: reconstruction error, energy metrics, compression ratios.

=== TUTORIAL: Understanding SVD-Hybrid Diagnostics ===

This module provides tools to analyze the quality of SVD-Hybrid compression
and merging. Good diagnostics help you:

1. **Tune parameters**: Find the right energy threshold and rank cap
2. **Verify quality**: Ensure reconstruction error is acceptable
3. **Measure compression**: Understand storage savings
4. **Debug issues**: Identify problematic parameters or tasks

=== KEY METRICS ===

1. **Reconstruction Error**: How well can we recover the original deltas?
   - absolute_error: ||original - reconstructed||
   - relative_error: ||error|| / ||original||
   - Target: <5% relative error is usually good

2. **Energy Retained**: What fraction of variance is captured?
   - 0.95 = 95% of variance in first k components
   - Higher is better but requires more storage

3. **Compression Ratio**: How much smaller is the compressed form?
   - original_size / compressed_size
   - Typical: 8-16x for 4-bit RTVQ

4. **Per-Parameter Statistics**: Which parameters compress well?
   - Some parameters naturally have low rank
   - Others need more components

=== DIAGNOSTIC OUTPUT ===

    {
        "summary": {
            "num_parameters": 50,
            "average_rank": 12.5,
            "average_energy_retained": 0.97,
            "average_reconstruction_error": 0.02,
            "average_compression_ratio": 10.5
        },
        "per_parameter": {
            "layer1.weight": {
                "k": 8,
                "energy_retained": 0.98,
                "mean_relative_error": 0.015
            },
            ...
        }
    }

=== EXAMPLE ===

    >>> from diagnostics import compute_all_diagnostics, print_diagnostics_summary
    >>> 
    >>> diagnostics = compute_all_diagnostics(
    ...     task_vectors, compressed, bases, masks, config
    ... )
    >>> print_diagnostics_summary(diagnostics)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from .rtvq import RTVQQuantizer, estimate_compression_ratio

if TYPE_CHECKING:
    from .config import SVDHybridConfig


def compute_reconstruction_error(
    original_delta: torch.Tensor,
    reconstructed_delta: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.
    
    Measures how well the compressed representation reconstructs the original.
    
    === ERROR METRICS ===
    
    - **absolute_error**: ||original - reconstructed||₂ (L2 norm of error)
    - **relative_error**: ||error||₂ / ||original||₂ (percentage of magnitude)
    - **max_absolute_error**: max(|error|) (worst element-wise error)
    - **mean_absolute_error**: mean(|error|) (average element-wise error)
    
    Args:
        original_delta: Original delta vector
        reconstructed_delta: Reconstructed delta vector
        
    Returns:
        Dictionary of error metrics
        
    Example:
        >>> metrics = compute_reconstruction_error(original, reconstructed)
        >>> if metrics["relative_error"] > 0.1:
        ...     print("Warning: >10% reconstruction error")
    """
    # Compute error vector
    error = original_delta - reconstructed_delta
    
    # Compute norms
    original_norm = original_delta.norm().item()
    error_norm = error.norm().item()
    
    # Relative error (avoid division by zero)
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


def compute_compression_statistics(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    compressed_all: Dict[str, Dict[str, Dict]],
    bases: Dict[str, Dict],
    config: "SVDHybridConfig"
) -> Dict:
    """
    Compute detailed compression statistics for scientific logging.
    
    This function analyzes the compression pipeline and produces detailed
    statistics showing before/after sizes, compression ratios per component,
    and which parts use FP16 vs quantized storage.
    
    === COMPRESSION COMPONENTS ===
    
    1. **High-energy coefficients (FP16)**:
       - First k coefficients per task
       - Stored in half-precision (16 bits = 2 bytes per value)
       - These capture the most important signal
    
    2. **Low-energy coefficients (RTVQ quantized)**:
       - Remaining D-k coefficients per task
       - Multi-stage residual quantization (e.g., 4-bit)
       - Each stage: indices + scale + zero_point
    
    3. **SVD Bases (shared across tasks)**:
       - U_high: [D × k] matrix in FP16
       - U_low: [D × (D-k)] matrix in FP16
       - Amortized across all tasks
    
    Args:
        task_vectors: Original task vectors
        compressed_all: Compressed coefficients for all parameters and tasks
        bases: SVD bases for all parameters
        config: SVDHybridConfig
        
    Returns:
        Detailed compression statistics dictionary
    """
    import math
    
    stats = {
        "original": {
            "total_bytes": 0,
            "per_task_bytes": {},
            "per_param_bytes": {}
        },
        "compressed": {
            "total_bytes": 0,
            "fp16_high_energy_bytes": 0,
            "rtvq_low_energy_bytes": 0,
            "svd_bases_bytes": 0,
            "per_task_bytes": {},
            "per_param_bytes": {}
        },
        "per_parameter": {},
        "summary": {}
    }
    
    num_tasks = len(task_vectors)
    num_bits = config.svd_low_bits
    num_stages = config.svd_rtvq_stages
    
    # Calculate original size
    for task_name, tv in task_vectors.items():
        task_bytes = sum(delta.numel() * 4 for delta in tv.values())  # FP32 = 4 bytes
        stats["original"]["per_task_bytes"][task_name] = task_bytes
        stats["original"]["total_bytes"] += task_bytes
    
    for param_name in task_vectors[list(task_vectors.keys())[0]].keys():
        param_bytes = sum(
            tv[param_name].numel() * 4 
            for tv in task_vectors.values() 
            if param_name in tv
        )
        stats["original"]["per_param_bytes"][param_name] = param_bytes
    
    # Calculate compressed size per parameter
    total_fp16_bytes = 0
    total_rtvq_bytes = 0
    total_bases_bytes = 0
    
    for param_name, basis in bases.items():
        if param_name not in compressed_all:
            continue
            
        param_stats = {
            "original_bytes": stats["original"]["per_param_bytes"].get(param_name, 0),
            "compressed_bytes": 0,
            "fp16_high_energy_bytes": 0,
            "rtvq_low_energy_bytes": 0,
            "svd_bases_bytes": 0,
            "k": 0,
            "D": 0,
            "compression_ratio": 0
        }
        
        basis_masked = basis.get("masked")
        if basis_masked is not None:
            k = basis_masked.get("k", 0)
            D = basis_masked.get("D", 0)
            param_stats["k"] = k
            param_stats["D"] = D
            
            # SVD bases size (shared across tasks, amortized)
            # U_high: [D × k] in FP16 = D * k * 2 bytes
            # U_low: [D × (D-k)] in FP16 = D * (D-k) * 2 bytes (but often truncated)
            u_high_numel = basis_masked.get("U_high", torch.empty(0)).numel()
            u_low_numel = basis_masked.get("U_low", torch.empty(0)).numel()
            bases_bytes = (u_high_numel + u_low_numel) * 2  # FP16 = 2 bytes
            param_stats["svd_bases_bytes"] = bases_bytes
            total_bases_bytes += bases_bytes
            
            # Per-task compressed coefficients
            compressed_params = compressed_all[param_name]
            for task_name, artifact in compressed_params.items():
                if artifact is None or artifact.get("masked") is None:
                    continue
                
                masked_artifact = artifact["masked"]
                
                # FP16 high-energy coefficients: k values × 2 bytes
                c_high = masked_artifact.get("c_high_fp16", torch.empty(0))
                fp16_bytes = c_high.numel() * 2
                param_stats["fp16_high_energy_bytes"] += fp16_bytes
                total_fp16_bytes += fp16_bytes
                
                # RTVQ low-energy coefficients
                # Each stage: quantized indices + scale (4 bytes) + zero_point (4 bytes)
                c_low_quant = masked_artifact.get("c_low_quant", {})
                payloads = c_low_quant.get("payloads", [])
                
                rtvq_bytes = 0
                for payload in payloads:
                    # Quantized indices: ceil(numel * num_bits / 8) bytes
                    q_indices = payload.get("quantized", torch.empty(0))
                    indices_bytes = math.ceil(q_indices.numel() * num_bits / 8)
                    # Scale and zero_point: 8 bytes per stage
                    overhead_bytes = 8
                    rtvq_bytes += indices_bytes + overhead_bytes
                
                param_stats["rtvq_low_energy_bytes"] += rtvq_bytes
                total_rtvq_bytes += rtvq_bytes
        
        param_stats["compressed_bytes"] = (
            param_stats["fp16_high_energy_bytes"] +
            param_stats["rtvq_low_energy_bytes"] +
            param_stats["svd_bases_bytes"]
        )
        
        if param_stats["original_bytes"] > 0:
            param_stats["compression_ratio"] = (
                param_stats["original_bytes"] / max(param_stats["compressed_bytes"], 1)
            )
        
        stats["per_parameter"][param_name] = param_stats
        stats["compressed"]["per_param_bytes"][param_name] = param_stats["compressed_bytes"]
    
    # Aggregate compressed stats
    stats["compressed"]["fp16_high_energy_bytes"] = total_fp16_bytes
    stats["compressed"]["rtvq_low_energy_bytes"] = total_rtvq_bytes
    stats["compressed"]["svd_bases_bytes"] = total_bases_bytes
    stats["compressed"]["total_bytes"] = total_fp16_bytes + total_rtvq_bytes + total_bases_bytes
    
    # Summary statistics
    original_total = stats["original"]["total_bytes"]
    compressed_total = stats["compressed"]["total_bytes"]
    
    stats["summary"] = {
        "original_size_mb": original_total / (1024 * 1024),
        "compressed_size_mb": compressed_total / (1024 * 1024),
        "overall_compression_ratio": original_total / max(compressed_total, 1),
        "fp16_fraction": total_fp16_bytes / max(compressed_total, 1),
        "rtvq_fraction": total_rtvq_bytes / max(compressed_total, 1),
        "bases_fraction": total_bases_bytes / max(compressed_total, 1),
        "num_parameters": len(stats["per_parameter"]),
        "num_tasks": num_tasks,
        "num_bits": num_bits,
        "num_stages": num_stages
    }
    
    return stats


def print_detailed_compression_report(
    compression_stats: Dict,
    config: "SVDHybridConfig",
    top_n_params: int = 5
) -> None:
    """
    Print a detailed scientific report of compression statistics.
    
    Shows mathematical breakdown of compression, before/after sizes,
    and which components use which precision.
    
    Args:
        compression_stats: Statistics from compute_compression_statistics
        config: SVDHybridConfig
        top_n_params: Number of top parameters to show in detail
    """
    summary = compression_stats.get("summary", {})
    original = compression_stats.get("original", {})
    compressed = compression_stats.get("compressed", {})
    per_param = compression_stats.get("per_parameter", {})
    
    # Pre-compute values for cleaner formatting
    energy_pct = int(config.svd_energy_threshold * 100)
    low_bits = config.svd_low_bits
    rtvq_stages = config.svd_rtvq_stages
    
    print(f"\n{'═'*80}")
    print(f"📊 DETAILED COMPRESSION ANALYSIS")
    print(f"{'═'*80}")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPRESSION MATHEMATICAL OVERVIEW                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 SVD DECOMPOSITION:                                                       │
│     For each parameter Δ (task vector delta):                               │
│                                                                              │
│     Stack N task deltas: T = [δ₁ | δ₂ | ... | δₙ]  ∈ ℝ^(D×N)               │
│     Compute SVD: T = U × Σ × Vᵀ                                             │
│                                                                              │
│     Split basis by energy:                                                  │
│       • U_high = U[:, :k]   (top k columns capturing {energy_pct}% energy)          │
│       • U_low = U[:, k:]    (remaining columns)                             │
│                                                                              │
│  📦 PROJECTION (per task):                                                  │
│     c_high = U_highᵀ × δ   →  k coefficients (high-energy)                 │
│     c_low = U_lowᵀ × δ     →  (D-k) coefficients (low-energy)              │
│                                                                              │
│  🗜️ QUANTIZATION:                                                           │
│     c_high → FP16 (16-bit float, 2 bytes/value)                            │
│     c_low  → {low_bits}-bit RTVQ × {rtvq_stages} stages                                          │
│                                                                              │
│  🔄 RTVQ (Residual Task Vector Quantization):                               │
│     Stage 1: q₁ = quantize(c_low), r₁ = c_low - dequant(q₁)                │
│     Stage 2: q₂ = quantize(r₁), r₂ = r₁ - dequant(q₂)                      │
│     ...                                                                      │
│     Reconstruction: c_low ≈ dequant(q₁) + dequant(q₂) + ...                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIZE COMPARISON (BEFORE vs AFTER)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📁 ORIGINAL (Uncompressed):                                                │
│     Storage: FP32 (32-bit float, 4 bytes per value)                        │
│     Total size: {original.get('total_bytes', 0) / (1024*1024):>10.4f} MB                                       │
│     Tasks: {summary.get('num_tasks', 0)} × Parameters: {summary.get('num_parameters', 0)}                                 │
│                                                                              │
│  📦 COMPRESSED:                                                              │
│     Total size: {compressed.get('total_bytes', 0) / (1024*1024):>10.4f} MB                                       │
│     ├─ FP16 high-energy coeffs: {compressed.get('fp16_high_energy_bytes', 0) / 1024:>10.2f} KB ({summary.get('fp16_fraction', 0)*100:>5.1f}%)     │
│     ├─ {config.svd_low_bits}-bit RTVQ low-energy:   {compressed.get('rtvq_low_energy_bytes', 0) / 1024:>10.2f} KB ({summary.get('rtvq_fraction', 0)*100:>5.1f}%)     │
│     └─ SVD bases (shared):     {compressed.get('svd_bases_bytes', 0) / 1024:>10.2f} KB ({summary.get('bases_fraction', 0)*100:>5.1f}%)     │
│                                                                              │
│  📈 COMPRESSION RATIO: {summary.get('overall_compression_ratio', 0):>6.2f}x                                        │
│     (Original / Compressed = {original.get('total_bytes', 0) / (1024*1024):.4f} MB / {compressed.get('total_bytes', 0) / (1024*1024):.4f} MB)                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRECISION BREAKDOWN BY COMPONENT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Component              │ Precision │ Bytes/Value │ Purpose                  │
│  ───────────────────────┼───────────┼─────────────┼─────────────────────────│
│  Original task vectors  │ FP32      │ 4.00        │ Full precision input    │
│  High-energy coeffs     │ FP16      │ 2.00        │ Important signal (top k)│
│  Low-energy coeffs      │ {config.svd_low_bits}-bit     │ {config.svd_low_bits/8:.2f}        │ Less important (D-k)    │
│  SVD bases (U_high)     │ FP16      │ 2.00        │ Shared projection basis │
│  SVD bases (U_low)      │ FP16      │ 2.00        │ Shared projection basis │
│                                                                              │
│  Note: {config.svd_low_bits}-bit RTVQ with {config.svd_rtvq_stages} stages uses {config.svd_low_bits}×{config.svd_rtvq_stages}={config.svd_low_bits*config.svd_rtvq_stages} bits effective per value          │
│        plus small overhead for scale/zero_point per stage                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Per-parameter breakdown (top N by original size)
    if per_param:
        sorted_params = sorted(
            per_param.items(),
            key=lambda x: x[1].get("original_bytes", 0),
            reverse=True
        )[:top_n_params]
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TOP {top_n_params} PARAMETERS BY SIZE (DETAILED BREAKDOWN)                │
├─────────────────────────────────────────────────────────────────────────────┤""")
        
        for param_name, pstats in sorted_params:
            orig_kb = pstats.get("original_bytes", 0) / 1024
            comp_kb = pstats.get("compressed_bytes", 0) / 1024
            ratio = pstats.get("compression_ratio", 0)
            k = pstats.get("k", 0)
            D = pstats.get("D", 0)
            fp16_kb = pstats.get("fp16_high_energy_bytes", 0) / 1024
            rtvq_kb = pstats.get("rtvq_low_energy_bytes", 0) / 1024
            
            # Truncate long parameter names
            display_name = param_name[:50] + "..." if len(param_name) > 50 else param_name
            
            print(f"""│                                                                              │
│  📍 {display_name:<55}│
│     Original: {orig_kb:>8.2f} KB → Compressed: {comp_kb:>8.2f} KB ({ratio:>5.2f}x)       │
│     SVD rank k={k}, dimension D={D:,}                                       │
│     ├─ FP16 high-energy: {fp16_kb:>8.2f} KB (top {k} coefficients)             │
│     └─ {config.svd_low_bits}-bit RTVQ:        {rtvq_kb:>8.2f} KB (remaining {D-k if D > k else 0} coefficients)     │""")
        
        print(f"""│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print(f"{'═'*80}\n")


def print_diagnostics_summary(diagnostics: Dict):
    """
    Print human-readable summary of diagnostics.
    
    Formats and prints the key metrics from a diagnostics dictionary
    for quick assessment of compression quality.
    
    Args:
        diagnostics: Diagnostics dictionary from compute_all_diagnostics
        
    Example output:
        ============================================================
        SVD-Hybrid Diagnostics Summary
        ============================================================
        
        Number of parameters: 50
        Average rank: 12.50 ± 5.20
        Average energy retained: 0.9650
        Average reconstruction error: 0.0234
        Average compression ratio: 10.50x
        
        ============================================================
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
