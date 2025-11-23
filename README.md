This module implements the three-phase SVD-Hybrid Merging approach for multi-task CLIP ViT-B/32 fine-tuned models (Cars, EuroSAT, DTD, SUN397) combining ideas from:
- TallMasks (localizing task information)
- Task Vector Quantization (TVQ) for memory-efficient storage of low-energy coefficients.

## Overview

Phases:

1. Basis Construction (Offline)
   - For each selected layer, stack task deltas (fine-tuned checkpoint weights minus base CLIP weights).
   - Run SVD → obtain U; choose rank k by cumulative energy threshold; split U into U_high (signal) and U_low (noise).

2. Task Compression (Per Task)
   - Project each delta: c_high = U_high^T τ, c_low = U_low^T τ.
   - Store c_high in FP16; quantize c_low with 2-bit RTVQ (TVQ).

3. Merging (Runtime)
   - Average all FP16 c_high vectors → c_avg_high.
   - Dequantize & average all c_low → c_avg_low.
   - Reconstruct merged delta: τ_final = U_high c_avg_high + U_low c_avg_low.
   - Apply to base weights to form merged model.

## Folder Structure

```
svd_hybrid_clip/
  config.py
  clip_loader.py
  task_collection.py
  svd_basis.py
  compression.py
  merging.py
  evaluation.py
  run_light_experiment.py
  tvq_adapter.py
  README.md
```

## Initial Task Set (Light)

Cars, EuroSAT, DTD, SUN397 (4 tasks):
- Use publicly available fine-tuned CLIP ViT-B/32 checkpoints (from Task Arithmetic or your tall_masks pipeline).
- Provide a list of checkpoint paths or HF hub IDs in `run_light_experiment.py`.

## Selecting Layers

To keep it light:
- Only a subset of transformer blocks (e.g., blocks 0, 6, 11).
- For each block: attention projection weights (qkv or in_proj), and MLP weights (fc1, fc2).

## Rank Selection

Default: retain ≥ 90% of cumulative energy, capped at k ≤ 32.

## Validation

Three metrics:
1. Reconstruction error per task per layer.
2. Accuracy on validation subsets for each dataset.
3. Compression ratio vs. raw FP16 deltas.

## Steps to Run

1. Gather base CLIP model (OpenAI CLIP) and fine-tuned checkpoints.
2. Edit `TASK_CHECKPOINTS` in `run_light_experiment.py`.
3. Run:
   ```bash
   python svd_hybrid_clip/run_light_experiment.py --data-root /path/to/datasets --output-root ./svd_out
   ```
4. Inspect logs for:
   - Chosen rank per layer
   - Reconstruction errors
   - Per-task and merged accuracy

## Integrating TVQ

Replace stub quantizer in `tvq_adapter.py` with actual functions from the `AIM-SKKU/TVQ` repo.

Example pattern (hypothetical):
```python
from tvq.rtvq import RTVQQuantizer
quantizer = RTVQQuantizer(bit_width=2)
q_obj = quantizer.quantize(c_low)      # returns structured quantized representation
c_low_deq = quantizer.dequantize(q_obj)
```

## Extending

- Add more tasks (8 → 14 → 20) by updating checkpoint list.
- Add weighted averaging (e.g., weight tasks by validation accuracy).
- Apply to all layers for improved synergy.
