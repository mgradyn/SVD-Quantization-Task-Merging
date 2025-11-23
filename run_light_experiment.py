import os
import argparse
import torch
from .config import SVDHybridConfig
from .clip_loader import load_clip_vit_b32, extract_visual_state_dict, filter_layers_for_light_run
from .task_collection import collect_task_deltas
from .svd_basis import build_all_bases
from .compression import compress_all_tasks
from .merging import merge_all_layers
from .evaluation import build_transform


TASK_CHECKPOINTS = [
    "/path/to/cars_clip_vitb32.pt",
    "/path/to/eurosat_clip_vitb32.pt",
    "/path/to/dtd_clip_vitb32.pt",
    "/path/to/sun397_clip_vitb32.pt"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="./svd_out")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = SVDHybridConfig()
    if args.device:
        cfg.device = args.device

    os.makedirs(args.output_root, exist_ok=True)

    # 1. Load base CLIP
    clip_model, preprocess = load_clip_vit_b32(device=cfg.device)
    base_visual_sd = extract_visual_state_dict(clip_model)

    # 2. Collect task deltas
    per_layer, task_order, raw_task_deltas = collect_task_deltas(
        base_visual_sd,
        TASK_CHECKPOINTS,
        filter_fn=lambda sd: filter_layers_for_light_run(sd, cfg.selected_blocks, cfg.layer_types),
        device=cfg.device
    )
    print(f"Collected deltas for layers: {list(per_layer.keys())}")
    print(f"Tasks: {task_order}")

    # 3. Build SVD bases
    bases = build_all_bases(per_layer, cfg)
    for lname, (_, _, meta) in bases.items():
        print(f"[Layer {lname}] k={meta['k']}, D={meta['D']}, N={meta['N']}")

    # 4. Compress each task (coeff projection + RTVQ)
    artifacts = compress_all_tasks(bases, per_layer, cfg.quant_bits_low)

    # 5. Merge layers
    merged_visual_sd = merge_all_layers(base_visual_sd, bases, artifacts)

    # 6. (Optional) Save artifacts
    torch.save({"bases": bases, "artifacts": artifacts, "task_order": task_order},
               os.path.join(args.output_root, "svd_hybrid_clip_artifacts.pt"))
    torch.save(merged_visual_sd, os.path.join(args.output_root, "merged_visual_sd.pt"))
    print("Saved merged visual state dict.")

    # 7. (Light) Reconstruction error check (sample first task & layer)
    for lname, layer_art_list in artifacts.items():
        # Reconstruct first task vector and compare with original delta
        U_high, U_low, meta = bases[lname]
        first_art = layer_art_list[0]
        c_high = first_art["c_high_fp16"].float()
        from .tvq_adapter import dequantize_low
        c_low = dequantize_low(first_art["c_low_quant"]).float()
        recon = U_high.float() @ c_high + U_low.float() @ c_low
        original_delta = per_layer[lname][0].float()
        rel_err = (recon - original_delta).norm() / original_delta.norm()
        print(f"[Recon Error] Layer={lname} Task={task_order[0]} rel_err={rel_err:.4f}")
        break

    print("Light experiment complete. Train/evaluate a small linear probe separately for proper accuracy comparison.")

if __name__ == "__main__":
    main()