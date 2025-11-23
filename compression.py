import torch
from typing import Dict
from .tvq_adapter import quantize_low

def project_coeffs(U_high: torch.Tensor, U_low: torch.Tensor, delta_flat: torch.Tensor):
    # Ensure consistent dtype (work in fp32 for projection accuracy)
    delta_f = delta_flat.float()
    c_high = (U_high.float().T @ delta_f)  # [k]
    c_low = (U_low.float().T @ delta_f)    # [D-k]
    return c_high, c_low

def compress_task_delta(U_high: torch.Tensor, U_low: torch.Tensor,
                        delta_flat: torch.Tensor,
                        quant_bits: int):
    c_high, c_low = project_coeffs(U_high, U_low, delta_flat)
    q_low = quantize_low(c_low, bit_width=quant_bits)
    return {
        "c_high_fp16": c_high.half(),
        "c_low_quant": q_low
    }

def compress_all_tasks(bases: Dict[str, tuple],
                       per_layer: Dict[str, list],
                       quant_bits: int):
    # per_layer[lname] : list of flattened delta vectors (one per task, order consistent)
    artifacts = {}
    for lname, (U_high, U_low, meta) in bases.items():
        layer_task_vecs = per_layer[lname]
        layer_art_list = []
        for delta_flat in layer_task_vecs:
            art = compress_task_delta(U_high, U_low, delta_flat, quant_bits)
            layer_art_list.append(art)
        artifacts[lname] = layer_art_list
    return artifacts