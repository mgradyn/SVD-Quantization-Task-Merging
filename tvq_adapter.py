import torch
from typing import Dict

# Placeholder RTVQ quantization. Replace with actual AIM-SKKU/TVQ API.

def quantize_low(vec: torch.Tensor, bit_width: int = 2) -> Dict:
    # Simple uniform quantization baseline; replace with RTVQ.
    vec_fp32 = vec.float()
    vmin, vmax = vec_fp32.min(), vec_fp32.max()
    spread = vmax - vmin + 1e-8
    levels = 2 ** bit_width
    normed = (vec_fp32 - vmin) / spread
    q = torch.clamp(torch.round(normed * (levels - 1)), 0, levels - 1).to(torch.uint8)
    return {
        "bit_width": bit_width,
        "min": vmin,
        "max": vmax,
        "data": q.cpu()
    }

def dequantize_low(qobj: Dict) -> torch.Tensor:
    levels = 2 ** qobj["bit_width"]
    q = qobj["data"].float() / (levels - 1)
    return q * (qobj["max"] - qobj["min"]) + qobj["min"]

# If TVQ is installed, auto-swap (pseudo):
try:
    import tvq  # hypothetical package root
    # Implement wrappers that call tvq's RTVQ routines
    # Provide a detection flag or override functions.
except ImportError:
    pass