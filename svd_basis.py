import torch
from typing import List, Dict, Tuple
from .config import SVDHybridConfig

def stack_and_center(vecs: List[torch.Tensor], center: bool) -> torch.Tensor:
    # vecs: list of [D]
    T = torch.stack(vecs, dim=1)  # [D, N]
    if center:
        mean = T.mean(dim=1, keepdim=True)
        T = T - mean
    return T

def choose_rank(S: torch.Tensor, threshold: float, max_rank: int) -> int:
    energy = (S ** 2)
    cum = torch.cumsum(energy, dim=0) / energy.sum()
    k = int((cum < threshold).sum().item()) + 1
    k = min(k, max_rank)
    return k

def compute_layer_basis(deltas_flat: List[torch.Tensor],
                        cfg: SVDHybridConfig,
                        device: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    T = stack_and_center(deltas_flat, cfg.center_task_matrix).to(device)
    # economy SVD
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    k = choose_rank(S, cfg.energy_threshold, cfg.max_rank)
    U_high = U[:, :k].contiguous()
    U_low = U[:, k:].contiguous()
    meta = {
        "k": k,
        "D": U.shape[0],
        "N": T.shape[1],
        "energy_threshold": cfg.energy_threshold,
        "singular_values": S[:k].cpu() if cfg.store_singular_values else None
    }
    return U_high, U_low, meta

def build_all_bases(per_layer: Dict[str, List[torch.Tensor]], cfg: SVDHybridConfig):
    bases = {}
    for lname, flat_list in per_layer.items():
        device = cfg.device
        U_high, U_low, meta = compute_layer_basis(flat_list, cfg, device)
        if cfg.fp_dtype == "fp16":
            U_high = U_high.half()
            U_low = U_low.half()
        bases[lname] = (U_high, U_low, meta)
    return bases