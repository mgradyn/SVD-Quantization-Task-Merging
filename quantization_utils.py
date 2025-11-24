"""
Core Task Vector Quantization (TVQ) utilities.

This module implements quantization methods following the TVQ (ICCV 2025) reference implementation.
Supports both absmax (signed) and asymmetric (min-max) quantization schemes.
"""
import torch
from typing import Tuple, Dict


def absmax_quantization(
    X: torch.Tensor,
    qbit: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Absmax quantization (symmetric, signed).
    
    Uses signed integer quantization based on absolute max value.
    For 8 bits: range [-128, 127] (int8)
    For 16 bits: range [-32768, 32767] (int16)
    
    Args:
        X: Input tensor to quantize
        qbit: Number of bits (8 or 16)
        
    Returns:
        Tuple of (quantized indices, scale factor)
    """
    if X.numel() == 0:
        return torch.zeros_like(X, dtype=torch.int8 if qbit == 8 else torch.int16), torch.tensor(1.0)
    
    # Compute scale based on absolute max
    abs_max = X.abs().max()
    
    if abs_max < 1e-10:
        scale = torch.tensor(1.0, device=X.device)
        quantized = torch.zeros_like(X, dtype=torch.int8 if qbit <= 8 else torch.int16)
        return quantized, scale
    
    # Determine quantization range
    if qbit == 8:
        qmin, qmax = -128, 127
        dtype = torch.int8
    elif qbit == 16:
        qmin, qmax = -32768, 32767
        dtype = torch.int16
    else:
        raise ValueError(f"Unsupported qbit: {qbit}, must be 8 or 16 for signed quantization")
    
    # Scale to fit in range
    scale = abs_max / qmax
    
    # Quantize
    quantized = torch.clamp(torch.round(X / scale), qmin, qmax)
    quantized = quantized.to(dtype)
    
    return quantized, scale


def asymmetric_quantization(
    X: torch.Tensor,
    qbit: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization (min-max, unsigned).
    
    Uses min-max scaling to unsigned integer range.
    For <=8 bits: uint8 range [0, 255]
    
    Follows reference TVQ implementation:
        scale = (qmax - qmin) / (X_max - X_min)
        zero_point = -round(scale * X_min)
        X_q = round(scale * X + zero_point).clamp(qmin, qmax)
    
    Args:
        X: Input tensor to quantize
        qbit: Number of bits (1-8 for uint8)
        
    Returns:
        Tuple of (quantized indices, scale, zero_point)
    """
    if X.numel() == 0:
        return torch.zeros_like(X, dtype=torch.uint8), torch.tensor(1.0), torch.tensor(0.0)
    
    # Compute min and max
    X_min = X.min()
    X_max = X.max()
    
    # Handle constant tensor
    if (X_max - X_min).abs() < 1e-8:
        scale = torch.tensor(1.0, device=X.device)
        zero_point = torch.tensor(0.0, device=X.device)
        quantized = torch.zeros_like(X, dtype=torch.uint8)
        return quantized, scale, zero_point
    
    # Quantization levels
    n_levels = 2 ** qbit
    qmin = 0
    qmax = n_levels - 1
    
    # Compute scale and zero_point following reference implementation
    scale = (qmax - qmin) / (X_max - X_min)
    zero_point = -torch.round(scale * X_min)
    
    # Clamp zero_point to valid range
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    # Quantize
    quantized = torch.round(scale * X + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    quantized = quantized.to(torch.uint8)
    
    return quantized, scale, zero_point


def dequantize_absmax(
    X_q: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize tensor quantized with absmax_quantization.
    
    Args:
        X_q: Quantized indices
        scale: Scale factor
        
    Returns:
        Dequantized tensor
    """
    return X_q.float() * scale


def dequantize_asymmetric(
    X_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize tensor quantized with asymmetric_quantization.
    
    Args:
        X_q: Quantized indices
        scale: Scale factor
        zero_point: Zero point
        
    Returns:
        Dequantized tensor
    """
    return (X_q.float() - zero_point) / scale


def quantization_error_check(
    X: torch.Tensor,
    X_q: torch.Tensor,
    scale: torch.Tensor,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Check quantization error for absmax quantization.
    
    Args:
        X: Original tensor
        X_q: Quantized tensor
        scale: Scale factor
        verbose: Whether to print error metrics
        
    Returns:
        Dictionary of error metrics
    """
    X_recon = dequantize_absmax(X_q, scale)
    
    # Compute L1 reconstruction error
    l1_error = (X - X_recon).abs().sum().item()
    l1_relative = l1_error / (X.abs().sum().item() + 1e-10)
    
    # Compute L2 error
    l2_error = (X - X_recon).pow(2).sum().sqrt().item()
    l2_relative = l2_error / (X.pow(2).sum().sqrt().item() + 1e-10)
    
    # Max error
    max_error = (X - X_recon).abs().max().item()
    
    metrics = {
        "l1_error": l1_error,
        "l1_relative": l1_relative,
        "l2_error": l2_error,
        "l2_relative": l2_relative,
        "max_error": max_error
    }
    
    if verbose:
        print(f"Quantization Error (absmax):")
        print(f"  L1 error: {l1_error:.6f} (relative: {l1_relative:.6f})")
        print(f"  L2 error: {l2_error:.6f} (relative: {l2_relative:.6f})")
        print(f"  Max error: {max_error:.6f}")
    
    return metrics


def quantization_error_check_asymmetric(
    X: torch.Tensor,
    X_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Check quantization error for asymmetric quantization.
    
    Args:
        X: Original tensor
        X_q: Quantized tensor
        scale: Scale factor
        zero_point: Zero point
        verbose: Whether to print error metrics
        
    Returns:
        Dictionary of error metrics
    """
    X_recon = dequantize_asymmetric(X_q, scale, zero_point)
    
    # Compute L1 reconstruction error
    l1_error = (X - X_recon).abs().sum().item()
    l1_relative = l1_error / (X.abs().sum().item() + 1e-10)
    
    # Compute L2 error
    l2_error = (X - X_recon).pow(2).sum().sqrt().item()
    l2_relative = l2_error / (X.pow(2).sum().sqrt().item() + 1e-10)
    
    # Max error
    max_error = (X - X_recon).abs().max().item()
    
    metrics = {
        "l1_error": l1_error,
        "l1_relative": l1_relative,
        "l2_error": l2_error,
        "l2_relative": l2_relative,
        "max_error": max_error
    }
    
    if verbose:
        print(f"Quantization Error (asymmetric):")
        print(f"  L1 error: {l1_error:.6f} (relative: {l1_relative:.6f})")
        print(f"  L2 error: {l2_error:.6f} (relative: {l2_relative:.6f})")
        print(f"  Max error: {max_error:.6f}")
    
    return metrics


def compute_compression_ratio(
    X: torch.Tensor,
    qbit: int,
    method: str = "asymmetric"
) -> float:
    """
    Compute theoretical compression ratio.
    
    Args:
        X: Original tensor
        qbit: Number of quantization bits
        method: "absmax" or "asymmetric"
        
    Returns:
        Compression ratio (original_bytes / compressed_bytes)
    """
    # Original size (assuming FP32)
    original_bytes = X.numel() * 4
    
    # Compressed size
    quantized_bytes = X.numel() * qbit / 8
    
    if method == "asymmetric":
        # scale (FP32) + zero_point (FP32)
        overhead_bytes = 8
    else:  # absmax
        # scale (FP32)
        overhead_bytes = 4
    
    compressed_bytes = quantized_bytes + overhead_bytes
    
    return original_bytes / compressed_bytes
