import torch
from typing import List, Dict

def load_finetuned_checkpoint(path: str, device="cpu"):
    """
    Expect a torch saved checkpoint with keys matching CLIP's visual backbone.
    For adaptation: if tall_masks uses a wrapper, adapt mapping here.
    """
    ckpt = torch.load(path, map_location=device)
    # If ckpt contains nested dict like {'model': state_dict}, adjust accordingly:
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
    return sd

def collect_task_deltas(base_visual_sd: Dict[str, torch.Tensor],
                        finetuned_paths: List[str],
                        filter_fn,
                        device="cpu"):
    """
    Returns:
      per_layer: layer_name -> list of flattened deltas (one per task)
      task_order: list of task identifiers
      raw_task_deltas: task_id -> {layer_name: delta_tensor (original shape)}
    """
    per_layer = {}
    raw_task_deltas = {}
    task_order = []
    for p in finetuned_paths:
        task_id = p.split("/")[-1]
        task_order.append(task_id)
        finetuned_sd = load_finetuned_checkpoint(p, device=device)
        # restrict to selected layers
        filtered = filter_fn(finetuned_sd)
        task_delta_layers = {}
        for lname, base_w in filter_fn(base_visual_sd).items():
            if lname not in filtered:
                continue
            w_task = filtered[lname].to(base_w.device)
            if w_task.shape != base_w.shape:
                continue
            delta = (w_task - base_w).detach()
            task_delta_layers[lname] = delta
            flat = delta.view(-1)
            per_layer.setdefault(lname, []).append(flat)
        raw_task_deltas[task_id] = task_delta_layers
    return per_layer, task_order, raw_task_deltas