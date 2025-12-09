"""
Core Task Vector Quantization (TVQ) utilities.

This module implements quantization methods following the TVQ (ICCV 2025) reference implementation.
Supports both absmax (signed) and asymmetric (min-max) quantization schemes.

=== TUTORIAL: Understanding Quantization ===

Quantization reduces the precision of floating-point values to lower-bit representations.
This significantly reduces storage and memory requirements at the cost of some accuracy loss.

Key Concepts:
-------------
1. **Quantization**: Converting high-precision floats (FP32) to low-bit integers (int8, uint8)
2. **Dequantization**: Reconstructing approximate floats from quantized integers
3. **Scale**: A multiplier that maps quantized values back to the original range
4. **Zero Point**: An offset used in asymmetric quantization to handle non-zero minimum values

Two Quantization Methods:
-------------------------
1. **Absmax (Symmetric)**: 
   - Uses signed integers (int8: -128 to 127)
   - Scale = max(|X|) / 127
   - X_quantized = round(X / scale)
   - Good for data centered around zero
   
2. **Asymmetric (Min-Max)**: 
   - Uses unsigned integers (uint8: 0 to 255)
   - Scale = 255 / (max(X) - min(X))
   - Zero_point = -round(scale * min(X))
   - X_quantized = round(scale * X + zero_point)
   - Better for data with non-zero mean or skewed distributions

Example Usage:
--------------
    >>> import torch
    >>> from quantization_utils import asymmetric_quantization, dequantize_asymmetric
    >>> 
    >>> # Create a tensor to quantize
    >>> X = torch.randn(100)
    >>> 
    >>> # Quantize to 8 bits
    >>> X_q, scale, zero_point = asymmetric_quantization(X, qbit=8)
    >>> 
    >>> # Reconstruct the original values
    >>> X_recon = dequantize_asymmetric(X_q, scale, zero_point)
    >>> 
    >>> # Check reconstruction error
    >>> error = (X - X_recon).abs().mean()
    >>> print(f"Mean absolute error: {error:.6f}")
"""

import torch
from typing import Tuple, Dict

# Constants for compression calculations
FLOAT32_BITS = 32  # Number of bits in a float32


def absmax_quantization(
    X: torch.Tensor,
    qbit: int = 8,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    s = (2**(qbit-1)-1)/torch.max(torch.abs(X))

    X_q = (s*X).round()
    if qbit<=8:
        dtype = torch.int8
    elif qbit==16:
        dtype = torch.int16
  
    return X_q.to(dtype), s


def asymmetric_quantization(
    X: torch.Tensor,
    qbit: int = 8,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  
    X_min = X.min()
    X_max = X.max()
    
    n_levels = 2 ** qbit
    qmin = 0
    qmax = n_levels - 1
   
    scale = (qmax - qmin) / (X_max - X_min) 
    zero_point =  -1* torch.round(scale * X_min)
    
    X_q = torch.round(scale * X + zero_point).clamp(qmin, qmax)

    if qbit<=8:
        dtype = torch.uint8
    elif qbit==16:
        dtype = torch.int16
    
    return X_q.to(dtype), scale, zero_point


def dequantize_absmax(
    X_q: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize tensor quantized with absmax_quantization.
    
    Reconstructs approximate original values by multiplying quantized integers by the scale.
    
    === HOW IT WORKS ===
    
    Formula: X_reconstructed = X_q * scale
    
    Since quantization involves rounding, the reconstructed values won't be exactly
    equal to the originals, but will be close (within one quantization step).
    
    Args:
        X_q: Quantized indices (int8 or int16)
        scale: Scale factor from quantization
        
    Returns:
        Dequantized tensor (float32) approximating original values
        
    Example:
        >>> X = torch.randn(100)
        >>> X_q, scale = absmax_quantization(X, qbit=8)
        >>> X_recon = dequantize_absmax(X_q, scale)
        >>> error = (X - X_recon).abs().max()
        >>> print(f"Max reconstruction error: {error:.6f}")
    """
    # Convert quantized integers to float and multiply by scale
    # X_q.float() converts int8/int16 to float32 for multiplication
    return X_q.float() * scale


def dequantize_asymmetric(
    X_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize tensor quantized with asymmetric_quantization.
    
    Reconstructs approximate original values using the inverse of the quantization formula.
    
    === HOW IT WORKS ===
    
    Quantization: X_q = round(scale * X + zero_point)
    Dequantization (inverse): X = (X_q - zero_point) / scale
    
    The reconstruction won't be exact due to rounding during quantization.
    
    Args:
        X_q: Quantized indices (uint8)
        scale: Scale factor from quantization
        zero_point: Zero point offset from quantization
        
    Returns:
        Dequantized tensor (float32) approximating original values
        
    Example:
        >>> X = torch.randn(100) + 5.0  # Non-zero mean data
        >>> X_q, scale, zero_point = asymmetric_quantization(X, qbit=8)
        >>> X_recon = dequantize_asymmetric(X_q, scale, zero_point)
        >>> print(f"Original mean: {X.mean():.3f}, Reconstructed mean: {X_recon.mean():.3f}")
    """
    # Apply inverse transformation:
    # 1. Convert uint8 to float32
    # 2. Subtract zero_point to shift back to original range
    # 3. Divide by scale to restore original magnitude
    return (X_q.float() - zero_point) / scale


def qunatization_error_check (original_state_dict, quantized_state_dict):
    accumulated_error = 0
    for key in original_state_dict.keys():
        weight_original = original_state_dict[key]
        weight_quantized = quantized_state_dict[key]
        if weight_quantized.dtype in [torch.int8]:
            if key + '_qscale' not in quantized_state_dict.keys():
                AssertionError('scale is missing for weight {}'.format(key))
            else:
                scale = quantized_state_dict[key + '_qscale']
            reconstructed_weight = weight_quantized.to(torch.float) / scale
        else:
            reconstructed_weight = weight_quantized
        error = weight_original - reconstructed_weight
        accumulated_error += torch.sum(torch.abs(error))#/torch.numel(error)
        # print(f'Error for weight {key}: {torch.max(torch.abs(error))}')
    print(f'accumuated Quantized error: {accumulated_error}')


def quantization_error_check_asymmetric (original_state_dict, quantized_state_dict):
    accumulated_error = 0
    for key in original_state_dict.keys():
        weight_original = original_state_dict[key]
        weight_quantized = quantized_state_dict[key]
        if weight_quantized.dtype in [torch.uint8]:
            if key + '_qscale' not in quantized_state_dict.keys():
                AssertionError('scale is missing for weight {}'.format(key))
            else:
                scale = quantized_state_dict[key + '_qscale']
                zero_point = quantized_state_dict[key + '_qzeropoint']

            reconstructed_weight = (weight_quantized.to(torch.float)  -zero_point.to(torch.float)) / scale
        else:
            reconstructed_weight = weight_quantized
        error = weight_original - reconstructed_weight
        accumulated_error += torch.sum(torch.abs(error))#/torch.numel(error)
        # print(f'Error for weight {key}: {torch.max(torch.abs(error))}')
    print(f'accumuated Quantized error: {accumulated_error}')