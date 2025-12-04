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


def absmax_quantization(
    X: torch.Tensor,
    qbit: int = 8,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Absmax quantization (symmetric, signed).
    
    Uses signed integer quantization based on absolute max value.
    For 8 bits: range [-128, 127] (int8)
    For 16 bits: range [-32768, 32767] (int16)
    
    === HOW IT WORKS ===
    
    1. Find the maximum absolute value in the tensor: abs_max = max(|X|)
    2. Compute scale to fit values in quantization range: scale = abs_max / 127 (for 8-bit)
    3. Quantize by dividing by scale: X_q = round(X / scale)
    4. Clamp to valid range: X_q = clamp(X_q, -128, 127)
    
    The scale factor is saved to enable dequantization later:
        X_reconstructed = X_q * scale
    
    === WHEN TO USE ===
    
    - Best for data that is roughly symmetric around zero
    - Common for neural network weight deltas (task vectors)
    - Uses signed integers, so negative values are represented directly
    
    Args:
        X: Input tensor to quantize (any shape, will be processed element-wise)
        qbit: Number of bits (8 or 16) - determines integer type and range
        verbose: If True, print educational tutorial-style progress messages
        
    Returns:
        Tuple of:
            - quantized: Tensor of quantized indices (int8 or int16)
            - scale: Scale factor for dequantization (single float value as tensor)
    
    Example:
        >>> X = torch.randn(100)  # Random values around 0
        >>> X_q, scale = absmax_quantization(X, qbit=8)
        >>> X_q.dtype
        torch.int8
        >>> X_q.min() >= -128 and X_q.max() <= 127
        True
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📚 TUTORIAL: Absmax (Symmetric) Quantization")
        print(f"{'='*70}")
        print(f"""
🎯 WHAT IS QUANTIZATION?
   Quantization compresses data by converting high-precision floats (32-bit)
   to low-precision integers ({qbit}-bit). This saves memory at the cost of
   some precision loss.
   
   ABSMAX (Symmetric) Quantization:
   • Uses SIGNED integers (can represent negative numbers directly)
   • Maps the range [-max|X|, +max|X|] to [{-2**(qbit-1)}, {2**(qbit-1)-1}]
   • Best for data centered around zero (like neural network weights)
""")
    
    # Handle edge case: empty tensor
    if X.numel() == 0:
        if verbose:
            print(f"   ⚠️ Empty tensor detected, returning zeros")
        return torch.zeros_like(X, dtype=torch.int8 if qbit == 8 else torch.int16), torch.tensor(1.0)
    
    if verbose:
        print(f"📊 INPUT ANALYSIS:")
        print(f"   └─ Tensor shape: {list(X.shape)}")
        print(f"   └─ Total elements: {X.numel():,}")
        print(f"   └─ Data type: {X.dtype}")
        print(f"   └─ Value range: [{X.min().item():.6f}, {X.max().item():.6f}]")
        print(f"   └─ Mean: {X.mean().item():.6f}, Std: {X.std().item():.6f}")
    
    # Step 1: Compute the absolute maximum value
    abs_max = X.abs().max()
    
    if verbose:
        print(f"\n🔢 STEP 1: Find Maximum Absolute Value")
        print(f"   └─ abs_max = max(|X|) = {abs_max.item():.6f}")
    
    # Handle edge case: all values are essentially zero
    if abs_max < 1e-10:
        if verbose:
            print(f"   ⚠️ All values near zero, returning zeros with scale=1.0")
        scale = torch.tensor(1.0, device=X.device)
        quantized = torch.zeros_like(X, dtype=torch.int8 if qbit <= 8 else torch.int16)
        return quantized, scale
    
    # Step 2: Determine the quantization range
    if qbit == 8:
        qmin, qmax = -128, 127
        dtype = torch.int8
    elif qbit == 16:
        qmin, qmax = -32768, 32767
        dtype = torch.int16
    else:
        raise ValueError(f"Unsupported qbit: {qbit}, must be 8 or 16 for signed quantization")
    
    if verbose:
        print(f"\n🔢 STEP 2: Determine Quantization Range")
        print(f"   └─ {qbit}-bit signed integers: [{qmin}, {qmax}]")
        print(f"   └─ Total quantization levels: {2**qbit}")
    
    # Step 3: Compute the scale factor
    scale = abs_max / qmax
    
    if verbose:
        print(f"\n🔢 STEP 3: Compute Scale Factor")
        print(f"   └─ scale = abs_max / qmax = {abs_max.item():.6f} / {qmax} = {scale.item():.8f}")
        print(f"   └─ Each integer step represents {scale.item():.8f} in original units")
    
    # Step 4: Quantize
    quantized = torch.clamp(torch.round(X / scale), qmin, qmax)
    quantized = quantized.to(dtype)
    
    if verbose:
        print(f"\n🔢 STEP 4: Apply Quantization")
        print(f"   └─ Formula: X_q = clamp(round(X / scale), {qmin}, {qmax})")
        print(f"   └─ Output dtype: {quantized.dtype}")
        print(f"   └─ Output range: [{quantized.min().item()}, {quantized.max().item()}]")
        
        # Validation: Check reconstruction quality
        X_reconstructed = quantized.float() * scale
        error = (X - X_reconstructed).abs()
        relative_error = error.sum().item() / (X.abs().sum().item() + 1e-10)
        max_error = error.max().item()
        
        print(f"""
   🔬 VALIDATION - RECONSTRUCTION QUALITY:
   ├─ Max absolute error: {max_error:.8f}
   ├─ Relative L1 error: {relative_error*100:.4f}%
   └─ Theoretical max error: {scale.item()/2:.8f} (half a quantization step)
""")
        
        # Quality assessment
        if relative_error < 0.01:
            print(f"   ✅ QUALITY CHECK PASSED: Excellent (<1% error)")
        elif relative_error < 0.05:
            print(f"   ✅ QUALITY CHECK PASSED: Good (<5% error)")
        elif relative_error < 0.10:
            print(f"   ⚠️ QUALITY CHECK WARNING: Moderate error (5-10%)")
        else:
            print(f"   ❌ QUALITY CHECK CONCERN: High error (>10%) - consider more bits")
        
        # Compression ratio
        original_bits = 32  # float32
        compression = original_bits / qbit
        print(f"""
   📦 COMPRESSION ACHIEVED:
   ├─ Original: {X.numel() * 4:,} bytes (float32)
   ├─ Quantized: {X.numel() * qbit // 8:,} bytes ({qbit}-bit) + 4 bytes (scale)
   └─ Compression ratio: {compression:.1f}x
""")
        print(f"{'='*70}\n")
    
    return quantized, scale


def asymmetric_quantization(
    X: torch.Tensor,
    qbit: int = 8,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization (min-max, unsigned).
    
    Uses min-max scaling to unsigned integer range.
    For <=8 bits: uint8 range [0, 255]
    
    === HOW IT WORKS ===
    
    This is the TVQ reference implementation formula:
    1. Find min and max: X_min = min(X), X_max = max(X)
    2. Compute scale: scale = (qmax - qmin) / (X_max - X_min) = 255 / (X_max - X_min)
    3. Compute zero point: zero_point = -round(scale * X_min)
    4. Quantize: X_q = round(scale * X + zero_point)
    5. Clamp to valid range: X_q = clamp(X_q, 0, 255)
    
    For dequantization:
        X_reconstructed = (X_q - zero_point) / scale
    
    === WHY USE ASYMMETRIC QUANTIZATION? ===
    
    - Handles data with any distribution (not just centered around zero)
    - Uses full range of unsigned integers efficiently
    - Better for data that is always positive or has a non-zero mean
    - Zero point allows precise representation of the original minimum value
    
    === LOWER BIT WIDTHS ===
    
    For qbit < 8, the quantization levels are reduced:
    - 4-bit: 16 levels (0-15)
    - 2-bit: 4 levels (0-3)
    - 1-bit: 2 levels (0-1) - essentially binarization
    
    Lower bits = higher compression but more quantization error.
    
    Args:
        X: Input tensor to quantize (any shape)
        qbit: Number of bits (1-8 for uint8 storage)
        verbose: If True, print educational tutorial-style progress messages
        
    Returns:
        Tuple of:
            - quantized: Tensor of quantized indices (uint8)
            - scale: Scale factor for dequantization
            - zero_point: Zero point offset for dequantization
    
    Example:
        >>> X = torch.randn(100) + 5.0  # Shifted data (not centered at 0)
        >>> X_q, scale, zero_point = asymmetric_quantization(X, qbit=8)
        >>> X_q.dtype
        torch.uint8
        >>> # Dequantize
        >>> X_recon = (X_q.float() - zero_point) / scale
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📚 TUTORIAL: Asymmetric (Min-Max) Quantization")
        print(f"{'='*70}")
        print(f"""
🎯 WHAT IS ASYMMETRIC QUANTIZATION?
   Unlike symmetric (absmax), asymmetric quantization:
   • Uses UNSIGNED integers (0 to {2**qbit - 1})
   • Maps [min(X), max(X)] to [0, {2**qbit - 1}]
   • Has a "zero point" to handle non-zero data ranges
   • Better for data that isn't centered around zero
   
   This is the method used in Task Vector Quantization (TVQ).
""")
    
    # Handle edge case: empty tensor
    if X.numel() == 0:
        if verbose:
            print(f"   ⚠️ Empty tensor detected, returning zeros")
        return torch.zeros_like(X, dtype=torch.uint8), torch.tensor(1.0), torch.tensor(0.0)
    
    if verbose:
        print(f"📊 INPUT ANALYSIS:")
        print(f"   └─ Tensor shape: {list(X.shape)}")
        print(f"   └─ Total elements: {X.numel():,}")
        print(f"   └─ Value range: [{X.min().item():.6f}, {X.max().item():.6f}]")
        print(f"   └─ Mean: {X.mean().item():.6f}, Std: {X.std().item():.6f}")
        
        # Check if data is centered
        mean = X.mean().item()
        if abs(mean) < 0.1 * X.std().item():
            print(f"   └─ Data appears centered around zero (absmax might work too)")
        else:
            print(f"   └─ Data is NOT centered around zero (good choice for asymmetric!)")
    
    # Step 1: Find min and max
    X_min = X.min()
    X_max = X.max()
    
    if verbose:
        print(f"\n🔢 STEP 1: Find Data Range")
        print(f"   └─ X_min = {X_min.item():.6f}")
        print(f"   └─ X_max = {X_max.item():.6f}")
        print(f"   └─ Range = {(X_max - X_min).item():.6f}")
    
    # Handle edge case: constant tensor
    if (X_max - X_min).abs() < 1e-8:
        if verbose:
            print(f"   ⚠️ Constant tensor detected (all values same), returning zeros")
        scale = torch.tensor(1.0, device=X.device)
        zero_point = torch.tensor(0.0, device=X.device)
        quantized = torch.zeros_like(X, dtype=torch.uint8)
        return quantized, scale, zero_point
    
    # Step 2: Determine quantization levels
    n_levels = 2 ** qbit
    qmin = 0
    qmax = n_levels - 1
    
    if verbose:
        print(f"\n🔢 STEP 2: Determine Quantization Levels")
        print(f"   └─ {qbit}-bit unsigned integers: [0, {qmax}]")
        print(f"   └─ Total quantization levels: {n_levels}")
    
    # Step 3: Compute scale
    scale = (qmax - qmin) / (X_max - X_min)
    
    if verbose:
        print(f"\n🔢 STEP 3: Compute Scale Factor")
        print(f"   └─ scale = {qmax} / (X_max - X_min)")
        print(f"   └─ scale = {qmax} / {(X_max - X_min).item():.6f} = {scale.item():.8f}")
    
    # Step 4: Compute zero point
    zero_point = -torch.round(scale * X_min)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    if verbose:
        print(f"\n🔢 STEP 4: Compute Zero Point")
        print(f"   └─ zero_point = -round(scale × X_min)")
        print(f"   └─ zero_point = -round({scale.item():.6f} × {X_min.item():.6f}) = {zero_point.item():.1f}")
        print(f"   └─ This means: X_min maps to approximately 0")
    
    # Step 5: Quantize
    quantized = torch.round(scale * X + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    quantized = quantized.to(torch.uint8)
    
    if verbose:
        print(f"\n🔢 STEP 5: Apply Quantization")
        print(f"   └─ Formula: X_q = clamp(round(scale × X + zero_point), 0, {qmax})")
        print(f"   └─ Output dtype: {quantized.dtype}")
        print(f"   └─ Output range: [{quantized.min().item()}, {quantized.max().item()}]")
        
        # Validation: Check reconstruction quality
        X_reconstructed = (quantized.float() - zero_point) / scale
        error = (X - X_reconstructed).abs()
        relative_error = error.sum().item() / (X.abs().sum().item() + 1e-10)
        max_error = error.max().item()
        
        print(f"""
   🔬 VALIDATION - RECONSTRUCTION QUALITY:
   ├─ To reconstruct: X_recon = (X_q - zero_point) / scale
   ├─ Max absolute error: {max_error:.8f}
   ├─ Relative L1 error: {relative_error*100:.4f}%
   └─ Theoretical max error: {(1.0/scale.item())/2:.8f} (half a quantization step)
""")
        
        # Verification test
        test_idx = 0
        original_val = X.flatten()[test_idx].item()
        quant_val = quantized.flatten()[test_idx].item()
        recon_val = (quant_val - zero_point.item()) / scale.item()
        
        print(f"   🧪 SAMPLE VERIFICATION (first element):")
        print(f"      Original: {original_val:.6f}")
        print(f"      Quantized: {quant_val}")
        print(f"      Reconstructed: {recon_val:.6f}")
        print(f"      Error: {abs(original_val - recon_val):.8f}")
        
        if abs(original_val - recon_val) < 0.1:
            print(f"      ✅ VERIFIED: Reconstruction is close to original")
        else:
            print(f"      ⚠️ NOTE: Some precision lost (expected with {qbit}-bit)")
        
        # Quality assessment
        if relative_error < 0.01:
            print(f"\n   ✅ QUALITY CHECK PASSED: Excellent (<1% error)")
        elif relative_error < 0.05:
            print(f"\n   ✅ QUALITY CHECK PASSED: Good (<5% error)")
        elif relative_error < 0.10:
            print(f"\n   ⚠️ QUALITY CHECK WARNING: Moderate error (5-10%)")
        else:
            print(f"\n   ❌ QUALITY CHECK CONCERN: High error (>10%) - consider more bits")
        
        # Compression ratio
        original_bits = 32  # float32
        compression = original_bits / qbit
        print(f"""
   📦 COMPRESSION ACHIEVED:
   ├─ Original: {X.numel() * 4:,} bytes (float32)
   ├─ Quantized: {X.numel() * qbit // 8:,} bytes ({qbit}-bit) + 8 bytes (scale, zp)
   └─ Compression ratio: {compression:.1f}x
""")
        print(f"{'='*70}\n")
    
    return quantized, scale, zero_point


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


def quantization_error_check(
    X: torch.Tensor,
    X_q: torch.Tensor,
    scale: torch.Tensor,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Check quantization error for absmax quantization.
    
    Computes various error metrics to assess quantization quality.
    Lower errors indicate better quantization fidelity.
    
    === ERROR METRICS EXPLAINED ===
    
    - **L1 Error**: Sum of absolute differences - measures total error magnitude
    - **L1 Relative**: L1 Error / sum(|X|) - normalized error, scale-independent
    - **L2 Error**: Euclidean distance (sqrt of sum of squared differences)
    - **L2 Relative**: L2 Error / ||X||_2 - normalized, emphasizes large errors
    - **Max Error**: Largest single-element error - worst-case error
    
    === TYPICAL VALUES ===
    
    For 8-bit quantization:
    - L1 relative: ~0.01-0.05 (1-5% total error)
    - L2 relative: ~0.01-0.05 (1-5% relative)
    - Max error: ~scale/2 (half a quantization step)
    
    For 4-bit quantization: errors are roughly 16x larger
    
    Args:
        X: Original tensor (float)
        X_q: Quantized tensor (int8/int16)
        scale: Scale factor from quantization
        verbose: If True, print formatted error metrics
        
    Returns:
        Dictionary of error metrics:
            - l1_error: Total L1 error
            - l1_relative: L1 error relative to tensor magnitude
            - l2_error: Total L2 error
            - l2_relative: L2 error relative to tensor magnitude
            - max_error: Maximum element-wise error
            
    Example:
        >>> X = torch.randn(1000)
        >>> X_q, scale = absmax_quantization(X, qbit=8)
        >>> metrics = quantization_error_check(X, X_q, scale, verbose=True)
        >>> if metrics['l1_relative'] > 0.1:
        ...     print("Warning: High quantization error!")
    """
    # First, reconstruct the original values from quantized representation
    X_recon = dequantize_absmax(X_q, scale)
    
    # Compute L1 reconstruction error (sum of absolute differences)
    # L1 error = Σ|X - X_recon|
    l1_error = (X - X_recon).abs().sum().item()
    
    # Relative L1 error: normalize by total magnitude of original tensor
    # This gives a percentage-like measure independent of tensor scale
    l1_relative = l1_error / (X.abs().sum().item() + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Compute L2 error (Euclidean distance between original and reconstructed)
    # L2 error = sqrt(Σ(X - X_recon)²)
    l2_error = (X - X_recon).pow(2).sum().sqrt().item()
    
    # Relative L2 error: normalize by L2 norm of original tensor
    l2_relative = l2_error / (X.pow(2).sum().sqrt().item() + 1e-10)
    
    # Maximum error: worst-case element-wise error
    # Useful for understanding the largest possible reconstruction error
    max_error = (X - X_recon).abs().max().item()
    
    # Package all metrics into a dictionary
    metrics = {
        "l1_error": l1_error,
        "l1_relative": l1_relative,
        "l2_error": l2_error,
        "l2_relative": l2_relative,
        "max_error": max_error
    }
    
    # Print formatted output if verbose mode is enabled
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
    
    Same error metrics as quantization_error_check, but for asymmetric quantization
    which uses both scale and zero_point for dequantization.
    
    Args:
        X: Original tensor (float)
        X_q: Quantized tensor (uint8)
        scale: Scale factor from quantization
        zero_point: Zero point from quantization
        verbose: If True, print formatted error metrics
        
    Returns:
        Dictionary of error metrics (same format as quantization_error_check):
            - l1_error, l1_relative, l2_error, l2_relative, max_error
            
    Example:
        >>> X = torch.randn(100)
        >>> X_q, scale, zp = asymmetric_quantization(X, qbit=4)
        >>> metrics = quantization_error_check_asymmetric(X, X_q, scale, zp, verbose=True)
    """
    # Reconstruct values using asymmetric dequantization
    X_recon = dequantize_asymmetric(X_q, scale, zero_point)
    
    # Compute L1 reconstruction error (sum of absolute differences)
    l1_error = (X - X_recon).abs().sum().item()
    l1_relative = l1_error / (X.abs().sum().item() + 1e-10)
    
    # Compute L2 error (Euclidean distance)
    l2_error = (X - X_recon).pow(2).sum().sqrt().item()
    l2_relative = l2_error / (X.pow(2).sum().sqrt().item() + 1e-10)
    
    # Maximum element-wise error
    max_error = (X - X_recon).abs().max().item()
    
    # Package metrics
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
    
    Calculates how much storage space is saved by quantization compared to
    storing full-precision (FP32) values.
    
    === HOW IT WORKS ===
    
    Compression Ratio = Original Size / Compressed Size
    
    Original Size (FP32): 
        num_elements × 4 bytes/element
    
    Compressed Size:
        - Quantized data: num_elements × qbit / 8 bytes
        - Metadata overhead: scale (4 bytes) + zero_point (4 bytes for asymmetric)
    
    === EXAMPLE COMPRESSION RATIOS ===
    
    For a tensor with 1000 elements:
    - 8-bit quantization: ~4x compression (4 bytes → 1 byte per element)
    - 4-bit quantization: ~8x compression (4 bytes → 0.5 bytes per element)
    - 2-bit quantization: ~16x compression (4 bytes → 0.25 bytes per element)
    
    Metadata overhead is negligible for large tensors but matters for small ones.
    
    Args:
        X: Original tensor (used to get element count)
        qbit: Number of quantization bits
        method: "asymmetric" (has zero_point overhead) or "absmax" (no zero_point)
        
    Returns:
        Compression ratio (larger is better - higher compression)
        
    Example:
        >>> X = torch.randn(10000)
        >>> ratio_8bit = compute_compression_ratio(X, qbit=8)
        >>> ratio_4bit = compute_compression_ratio(X, qbit=4)
        >>> print(f"8-bit: {ratio_8bit:.1f}x, 4-bit: {ratio_4bit:.1f}x")
        8-bit: 4.0x, 4-bit: 8.0x
    """
    # Calculate original size in bytes
    # FP32 = 32 bits = 4 bytes per element
    original_bytes = X.numel() * 4
    
    # Calculate compressed size in bytes
    # Quantized data: each element uses qbit bits → qbit/8 bytes
    quantized_bytes = X.numel() * qbit / 8
    
    # Add metadata overhead for dequantization
    if method == "asymmetric":
        # scale (FP32 = 4 bytes) + zero_point (FP32 = 4 bytes)
        overhead_bytes = 8
    else:  # absmax
        # Only scale (FP32 = 4 bytes)
        overhead_bytes = 4
    
    # Total compressed size
    compressed_bytes = quantized_bytes + overhead_bytes
    
    # Compression ratio = how many times smaller the compressed version is
    return original_bytes / compressed_bytes
