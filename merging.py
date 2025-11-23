import torch
from typing import List, Dict
from .tvq_adapter import dequantize_low

def average_high(c_high_list: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([v.float() for v in c_high_list], dim=0)
    return stacked.mean(dim=0)

def average_low(q_low_list: List[Dict]) -> torch.Tensor:
    deq = [dequantize_low(q).float() for q in q_low_list]
    stacked = torch.stack(deq, dim=0)
    return stacked.mean(dim=0)

def reconstruct(U_high: torch.Tensor, U_low: torch.Tensor,
                c_avg_high: torch.Tensor, c_avg_low: torch.Tensor) -> torch.Tensor:
    # U_high: [D, k], c_avg_high: [k]; similar for low
    part_high = U_high.float() @ c_avg_high.float()
    part_low = U_low.float() @ c_avg_low.float()
    return part_high + part_low

def merge_layer(basis_tuple, layer_artifacts: List[Dict]):
    U_high, U_low, meta = basis_tuple
    c_high_list = [art["c_high_fp16"] for art in layer_artifacts]
    q_low_list = [art["c_low_quant"] for art in layer_artifacts]
    c_avg_high = average_high(c_high_list)
    c_avg_low = average_low(q_low_list)
    merged_flat = reconstruct(U_high, U_low, c_avg_high, c_avg_low)
    return merged_flat

def merge_all_layers(base_visual_sd: Dict[str, torch.Tensor],
                     bases: Dict[str, tuple],
                     artifacts: Dict[str, List[Dict]]):
    merged_sd = {}
    for name, base_w in base_visual_sd.items():
        if name in bases:
            merged_flat = merge_layer(bases[name], artifacts[name])
            merged_w = base_w + merged_flat.view(base_w.shape).to(base_w.dtype)
            merged_sd[name] = merged_w
        else:
            merged_sd[name] = base_w
    return merged_sd