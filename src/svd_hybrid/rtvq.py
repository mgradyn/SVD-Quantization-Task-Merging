"""
Advanced Residual Task Vector Quantization (RTVQ) with multi-stage refinement.
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
    
    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits for quantization
        per_channel: Whether to quantize per-channel (not implemented)
        
    Returns:
        Tuple of (quantized indices, scale, zero_point)
    """
    if tensor.numel() == 0:
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.tensor(1.0), torch.tensor(0.0)
    
    # Compute range
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Avoid division by zero
    if max_val - min_val < 1e-8:
        scale = torch.tensor(1.0, device=tensor.device)
        zero_point = torch.tensor(0.0, device=tensor.device)
        quantized = torch.zeros_like(tensor, dtype=torch.uint8)
        return quantized, scale, zero_point
    
    # Compute scale and zero point
    n_levels = 2 ** num_bits
    scale = (max_val - min_val) / (n_levels - 1)
    zero_point = min_val
    
    # Quantize
    normalized = (tensor - zero_point) / scale
    quantized = torch.clamp(torch.round(normalized), 0, n_levels - 1)
    quantized = quantized.to(torch.uint8)
    
    return quantized, scale, zero_point


def asymmetric_dequantization(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize asymmetrically quantized tensor.
    
    Args:
        quantized: Quantized indices
        scale: Scale factor
        zero_point: Zero point
        
    Returns:
        Dequantized tensor
    """
    dequantized = quantized.float() * scale + zero_point
    return dequantized


def multistage_residual_quantization(
    tensor: torch.Tensor,
    num_bits: int = 4,
    num_stages: int = 2
) -> List[Dict]:
    """
    Multi-stage residual quantization.
    
    Each stage quantizes the residual from previous stages.
    
    Args:
        tensor: Input tensor to quantize
        num_bits: Number of bits per stage
        num_stages: Number of refinement stages
        
    Returns:
        List of quantization payloads, one per stage
    """
    if tensor.numel() == 0:
        return []
    
    residual = tensor.clone()
    payloads = []
    
    for stage in range(num_stages):
        # Quantize current residual
        q_indices, scale, zero_point = asymmetric_quantization(residual, num_bits)
        
        # Dequantize to compute next residual
        dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
        
        # Store payload
        payload = {
            "stage": stage,
            "quantized": q_indices.cpu(),
            "scale": scale.cpu(),
            "zero_point": zero_point.cpu(),
            "residual_norm": residual.norm().item()
        }
        payloads.append(payload)
        
        # Compute residual for next stage
        residual = residual - dequantized
    
    return payloads


def multistage_residual_dequantization(payloads: List[Dict], device: str = "cpu") -> torch.Tensor:
    """
    Dequantize multi-stage residual quantized tensor.
    
    Args:
        payloads: List of quantization payloads from multistage_residual_quantization
        device: Device to place result on
        
    Returns:
        Reconstructed tensor
    """
    if not payloads:
        return torch.tensor([], device=device)
    
    # Start with first stage
    result = torch.zeros_like(payloads[0]["quantized"].float(), device=device)
    
    for payload in payloads:
        q_indices = payload["quantized"].to(device)
        scale = payload["scale"].to(device)
        zero_point = payload["zero_point"].to(device)
        
        # Dequantize and add to result
        dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
        result = result + dequantized
    
    return result


class RTVQQuantizer:
    """
    RTVQ Quantizer with configurable bits and stages.
    """
    
    def __init__(self, num_bits: int = 4, num_stages: int = 2):
        """
        Initialize quantizer.
        
        Args:
            num_bits: Number of bits per stage
            num_stages: Number of refinement stages
        """
        self.num_bits = num_bits
        self.num_stages = num_stages
    
    def quantize(self, tensor: torch.Tensor) -> Dict:
        """
        Quantize tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Quantization payload dictionary
        """
        payloads = multistage_residual_quantization(
            tensor, 
            num_bits=self.num_bits,
            num_stages=self.num_stages
        )
        
        return {
            "payloads": payloads,
            "num_bits": self.num_bits,
            "num_stages": self.num_stages,
            "original_shape": tensor.shape,
            "original_dtype": str(tensor.dtype)
        }
    
    def dequantize(self, quantized_obj: Dict, device: str = "cpu") -> torch.Tensor:
        """
        Dequantize tensor.
        
        Args:
            quantized_obj: Quantization payload from quantize()
            device: Device to place result on
            
        Returns:
            Reconstructed tensor
        """
        result = multistage_residual_dequantization(
            quantized_obj["payloads"],
            device=device
        )
        
        # Reshape if needed
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
