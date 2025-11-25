"""
Advanced Residual Task Vector Quantization (RTVQ) with multi-stage refinement.

=== TUTORIAL: Understanding RTVQ ===

RTVQ (Residual Task Vector Quantization) is a multi-stage quantization technique
that progressively refines the reconstruction by quantizing residuals.

=== THE PROBLEM WITH SINGLE-STAGE QUANTIZATION ===

Standard quantization maps continuous values to discrete levels:
    X_q = quantize(X)
    X_recon = dequantize(X_q)
    error = X - X_recon

With 4 bits (16 levels), the error can be significant. Instead of just accepting
this error, RTVQ quantizes it in subsequent stages.

=== HOW RTVQ WORKS ===

Stage 1: Quantize original values
    X_q1 = quantize(X)
    X_recon1 = dequantize(X_q1)
    residual1 = X - X_recon1

Stage 2: Quantize the residual
    X_q2 = quantize(residual1)
    X_recon2 = dequantize(X_q2)
    residual2 = residual1 - X_recon2

Final reconstruction:
    X_final = X_recon1 + X_recon2

Each stage reduces the error by capturing what previous stages missed.

=== BENEFITS ===

1. **Better accuracy**: Multi-stage captures more detail than single-stage
2. **Same bits**: Each stage uses same bit width (e.g., 4-bit)
3. **Diminishing returns**: Most value from 2-3 stages
4. **Flexible**: Can choose accuracy/compression tradeoff

=== TYPICAL SETTINGS ===

- 2 stages: Good balance (default)
- 3 stages: Higher quality at cost of more storage
- 4+ stages: Diminishing returns, rarely needed

=== EXAMPLE ===

    >>> from rtvq import RTVQQuantizer
    >>> 
    >>> # Create quantizer with 4 bits, 2 stages
    >>> quantizer = RTVQQuantizer(num_bits=4, num_stages=2)
    >>> 
    >>> # Quantize a tensor
    >>> coefficients = torch.randn(100)
    >>> quantized = quantizer.quantize(coefficients)
    >>> 
    >>> # Reconstruct
    >>> reconstructed = quantizer.dequantize(quantized)
    >>> 
    >>> # Check error
    >>> error = (coefficients - reconstructed).abs().mean()
"""

import torch
from typing import Dict, List, Tuple, Optional


def asymmetric_quantization(
    tensor: torch.Tensor,
    num_bits: int = 4,
    per_channel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric uniform quantization.
    
    Maps floating-point values to unsigned integers using min-max scaling.
    This is the core quantization used in RTVQ for each stage.
    
    === FORMULA ===
    
    Given tensor with values in range [X_min, X_max]:
    
    scale = (qmax - qmin) / (X_max - X_min)
    zero_point = -round(scale * X_min)
    X_q = round(scale * X + zero_point)
    
    Where qmin=0, qmax=2^num_bits - 1 (e.g., 0-15 for 4-bit)
    
    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits for quantization (1-8)
        per_channel: Whether to quantize per-channel (not implemented)
        
    Returns:
        Tuple of:
            - quantized: uint8 tensor of quantized indices
            - scale: Scale factor for dequantization
            - zero_point: Zero point for dequantization
    """
    # Handle empty tensor
    if tensor.numel() == 0:
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.tensor(1.0), torch.tensor(0.0)
    
    # Find range of values
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Handle constant tensor (avoid division by zero)
    if max_val - min_val < 1e-8:
        scale = torch.tensor(1.0, device=tensor.device)
        zero_point = torch.tensor(0.0, device=tensor.device)
        quantized = torch.zeros_like(tensor, dtype=torch.uint8)
        return quantized, scale, zero_point
    
    # Compute scale and zero point following TVQ reference implementation:
    # scale = (qmax - qmin) / (X_max - X_min)
    # zero_point = -round(scale * X_min)
    # X_q = round(scale * X + zero_point).clamp(qmin, qmax)
    n_levels = 2 ** num_bits
    qmin = 0
    qmax = n_levels - 1
    
    scale = (qmax - qmin) / (max_val - min_val)
    zero_point = -torch.round(scale * min_val)
    
    # Clamp zero_point to valid range
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    # Quantize: scale, shift, round, clamp
    quantized = torch.round(scale * tensor + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    quantized = quantized.to(torch.uint8)
    
    return quantized, scale, zero_point


def asymmetric_dequantization(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize asymmetrically quantized tensor.
    
    Inverts the quantization formula to recover approximate original values.
    
    === FORMULA ===
    
    Quantization: X_q = round(scale * X + zero_point)
    Dequantization: X_recon = (X_q - zero_point) / scale
    
    Args:
        quantized: Quantized indices (uint8)
        scale: Scale factor from quantization
        zero_point: Zero point from quantization
        
    Returns:
        Dequantized tensor (float32)
    """
    # Reverse the quantization transformation
    dequantized = (quantized.float() - zero_point) / scale
    return dequantized


def multistage_residual_quantization(
    tensor: torch.Tensor,
    num_bits: int = 4,
    num_stages: int = 2
) -> List[Dict]:
    """
    Multi-stage residual quantization.
    
    Each stage quantizes the residual (error) from previous stages,
    progressively improving reconstruction accuracy.
    
    === ALGORITHM ===
    
    Stage 1:
        q1, s1, zp1 = quantize(tensor)
        recon1 = dequantize(q1, s1, zp1)
        residual = tensor - recon1
        
    Stage 2:
        q2, s2, zp2 = quantize(residual)
        recon2 = dequantize(q2, s2, zp2)
        residual = residual - recon2
        
    ... (continue for more stages)
    
    Final reconstruction = recon1 + recon2 + ...
    
    === STORAGE ===
    
    Each stage stores:
    - quantized: uint8 indices
    - scale: float32 scale factor
    - zero_point: float32 zero point
    - residual_norm: norm of remaining residual (for diagnostics)
    
    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits per stage (e.g., 4)
        num_stages: Number of refinement stages (e.g., 2)
        
    Returns:
        List of quantization payloads, one per stage
        
    Example:
        >>> x = torch.randn(100)
        >>> payloads = multistage_residual_quantization(x, num_bits=4, num_stages=2)
        >>> len(payloads)
        2
        >>> payloads[0].keys()
        dict_keys(['stage', 'quantized', 'scale', 'zero_point', 'residual_norm'])
    """
    # Handle empty tensor
    if tensor.numel() == 0:
        return []
    
    # Start with the original tensor as the residual
    residual = tensor.clone()
    payloads = []
    
    for stage in range(num_stages):
        # Quantize current residual
        q_indices, scale, zero_point = asymmetric_quantization(residual, num_bits)
        
        # Dequantize to compute next residual
        dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
        
        # Store this stage's payload
        payload = {
            "stage": stage,
            "quantized": q_indices.cpu(),      # Move to CPU for storage
            "scale": scale.cpu(),
            "zero_point": zero_point.cpu(),
            "residual_norm": residual.norm().item()  # For diagnostics
        }
        payloads.append(payload)
        
        # Compute residual for next stage (what this stage didn't capture)
        residual = residual - dequantized
    
    return payloads


def multistage_residual_dequantization(payloads: List[Dict], device: str = "cpu") -> torch.Tensor:
    """
    Dequantize multi-stage residual quantized tensor.
    
    Reconstructs the original tensor by summing dequantized values from
    all stages. Each stage contributes its reconstruction of the residual.
    
    === FORMULA ===
    
    X_reconstructed = Σ dequantize(stage_i) for all stages
                    = recon_stage1 + recon_stage2 + ...
    
    Args:
        payloads: List of quantization payloads from multistage_residual_quantization
        device: Device to place result on ("cpu" or "cuda")
        
    Returns:
        Reconstructed tensor (sum of all stage reconstructions)
        
    Example:
        >>> payloads = multistage_residual_quantization(x, num_bits=4, num_stages=2)
        >>> x_recon = multistage_residual_dequantization(payloads)
        >>> error = (x - x_recon).abs().mean()
    """
    # Handle empty payload list
    if not payloads:
        return torch.tensor([], device=device)
    
    # Initialize result with zeros (same shape as first stage's quantized data)
    result = torch.zeros_like(payloads[0]["quantized"].float(), device=device)
    
    # Sum dequantized values from all stages
    for payload in payloads:
        q_indices = payload["quantized"].to(device)
        scale = payload["scale"].to(device)
        zero_point = payload["zero_point"].to(device)
        
        # Dequantize this stage and add to result
        dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
        result = result + dequantized
    
    return result


class RTVQQuantizer:
    """
    RTVQ Quantizer with configurable bits and stages.
    
    A convenient wrapper class for multi-stage residual quantization.
    Provides a clean interface for quantizing and dequantizing tensors
    with consistent settings.
    
    === USAGE ===
    
        >>> quantizer = RTVQQuantizer(num_bits=4, num_stages=2)
        >>> 
        >>> # Quantize
        >>> payload = quantizer.quantize(tensor)
        >>> 
        >>> # Later, dequantize
        >>> tensor_recon = quantizer.dequantize(payload)
    
    === TYPICAL CONFIGURATIONS ===
    
    - Low quality, high compression: num_bits=2, num_stages=2
    - Balanced (default): num_bits=4, num_stages=2
    - High quality: num_bits=4, num_stages=3
    - Very high quality: num_bits=6, num_stages=3
    
    Attributes:
        num_bits: Bits per quantization stage
        num_stages: Number of residual refinement stages
    """
    
    def __init__(self, num_bits: int = 4, num_stages: int = 2):
        """
        Initialize quantizer.
        
        Args:
            num_bits: Number of bits per stage (e.g., 4 = 16 levels)
            num_stages: Number of refinement stages (e.g., 2)
        """
        self.num_bits = num_bits
        self.num_stages = num_stages
    
    def quantize(self, tensor: torch.Tensor) -> Dict:
        """
        Quantize tensor using multi-stage residual quantization.
        
        Args:
            tensor: Input tensor (any shape)
            
        Returns:
            Quantization payload dictionary containing:
                - payloads: List of per-stage quantization data
                - num_bits: Bit width used
                - num_stages: Number of stages used
                - original_shape: Shape of input tensor
                - original_dtype: Data type of input tensor
        """
        # Perform multi-stage quantization
        payloads = multistage_residual_quantization(
            tensor, 
            num_bits=self.num_bits,
            num_stages=self.num_stages
        )
        
        # Package with metadata for later dequantization
        return {
            "payloads": payloads,
            "num_bits": self.num_bits,
            "num_stages": self.num_stages,
            "original_shape": tensor.shape,
            "original_dtype": str(tensor.dtype)
        }
    
    def dequantize(self, quantized_obj: Dict, device: str = "cpu") -> torch.Tensor:
        """
        Dequantize tensor from quantization payload.
        
        Args:
            quantized_obj: Quantization payload from quantize()
            device: Device to place result on
            
        Returns:
            Reconstructed tensor with original shape
        """
        # Sum dequantized values from all stages
        result = multistage_residual_dequantization(
            quantized_obj["payloads"],
            device=device
        )
        
        # Reshape if needed (payloads store flat data)
        if "original_shape" in quantized_obj:
            result = result.view(quantized_obj["original_shape"])
        
        return result


def compute_quantization_error(
    original: torch.Tensor,
    quantized_obj: Dict,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute quantization error metrics.
    
    Args:
        original: Original tensor
        quantized_obj: Quantization payload
        device: Device for computation
        
    Returns:
        Dictionary of error metrics
    """
    quantizer = RTVQQuantizer(
        num_bits=quantized_obj["num_bits"],
        num_stages=quantized_obj["num_stages"]
    )
    
    reconstructed = quantizer.dequantize(quantized_obj, device)
    
    error = original - reconstructed
    
    relative_error = error.norm() / original.norm() if original.norm() > 1e-10 else 0
    
    return {
        "absolute_error": error.norm().item(),
        "relative_error": relative_error.item(),
        "max_absolute_error": error.abs().max().item(),
        "mean_absolute_error": error.abs().mean().item()
    }


def estimate_compression_ratio(
    original: torch.Tensor,
    quantized_obj: Dict
) -> float:
    """
    Estimate compression ratio.
    
    Args:
        original: Original tensor
        quantized_obj: Quantization payload
        
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    # Original size in bytes (assuming FP32)
    original_size = original.numel() * 4  # 4 bytes per float32
    
    # Compressed size
    num_stages = quantized_obj["num_stages"]
    num_bits = quantized_obj["num_bits"]
    
    # Each stage: quantized indices + scale + zero_point
    indices_size = original.numel() * num_bits / 8  # bits to bytes
    overhead_size = 8 * num_stages  # scale and zero_point per stage (float32)
    
    compressed_size = indices_size * num_stages + overhead_size
    
    ratio = original_size / max(compressed_size, 1)
    
    return ratio
