import torch
from typing import Dict, List, Tuple, Optional

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

def asymmetric_dequantization(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    # Reverse the quantization transformation
    dequantized = (quantized.float() - zero_point) / scale
    return dequantized


def multistage_residual_quantization(
    tensor: torch.Tensor,
    num_bits: int = 4,
    num_stages: int = 2,
    verbose: bool = False
) -> List[Dict]:    
    # Handle empty tensor
    if tensor.numel() == 0:
        if verbose:
            print(f"   ⚠️ Empty tensor, returning empty result")
        return []
    
    # Start with the original tensor as the residual
    residual = tensor.clone()
    payloads = []

    for stage in range(num_stages):
        residual_norm_before = residual.norm().item()
        
        # Quantize current residual
        q_indices, scale, zero_point = asymmetric_quantization(residual, num_bits)
        
        # Dequantize to compute next residual
        dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
        
        # Compute residual for next stage (what this stage didn't capture)
        new_residual = residual - dequantized
        residual_norm_after = new_residual.norm().item()
        
        # Store this stage's payload
        payload = {
            "stage": stage,
            "quantized": q_indices.cpu(),      # Move to CPU for storage
            "scale": scale.cpu(),
            "zero_point": zero_point.cpu(),
            "residual_norm": residual_norm_before  # For diagnostics
        }
        payloads.append(payload)

        # Update residual for next stage
        residual = new_residual
    
    
    return payloads


def multistage_residual_dequantization(payloads: List[Dict], device: str = "cpu") -> torch.Tensor:
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
    def __init__(self, num_bits: int = 4, num_stages: int = 2):
        self.num_bits = num_bits
        self.num_stages = num_stages
    
    def quantize(self, tensor: torch.Tensor) -> Dict:
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
        # Sum dequantized values from all stages
        result = multistage_residual_dequantization(
            quantized_obj["payloads"],
            device=device
        )
        
        # Reshape if needed (payloads store flat data)
        if "original_shape" in quantized_obj:
            result = result.view(quantized_obj["original_shape"])
        
        return result


def estimate_compression_ratio(
    original: torch.Tensor,
    quantized_obj: Dict
) -> float:
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
