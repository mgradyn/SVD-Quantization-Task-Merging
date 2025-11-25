# SVD-Hybrid Merging Method

## Quick Tutorial: What is SVD-Hybrid?

SVD-Hybrid is a technique for combining multiple fine-tuned machine learning models into a single model that can perform well on all their tasks. Think of it like this:

**Problem**: You have 8 different models, each specialized for one task (cars, textures, scenes, etc.). Running all 8 models is slow and expensive.

**Solution**: SVD-Hybrid merges them into ONE model that handles ALL tasks.

### How It Works (Simple Version)

1. **Extract what each model learned**: For each fine-tuned model, compute the difference from the base model. This difference is called a "task vector".

2. **Find common patterns**: Use SVD (a math technique) to find the most important directions that explain all task vectors.

3. **Keep important stuff, compress the rest**: Important parts stay in high precision (FP16). Less important parts get heavily compressed (4-bit).

4. **Average everything together**: Combine all tasks with weights (equal, performance-based, or clustered).

5. **Create merged model**: Add the merged task vector to the base model.

### Visual Overview

```
Fine-tuned Models           Task Vectors              SVD Compression
[Cars Model]     ->    [Cars Delta]      ┐
[DTD Model]      ->    [DTD Delta]       ├─> [High Energy (FP16)]
[EuroSAT Model]  ->    [EuroSAT Delta]   │   [Low Energy (4-bit)]
[SUN397 Model]   ->    [SUN397 Delta]    ┘
                            │
                            ▼
                    Weighted Average
                            │
                            ▼
                [Base Model] + [Merged Delta] = [Merged Model]
```

## Overview

SVD-Hybrid is an advanced model merging technique that combines:

1. **Tall Mask localization**: Uses binary masks to identify task-specific parameters
2. **SVD-based compression**: Decomposes task vectors into high and low energy subspaces
3. **Residual quantization**: Applies multi-stage quantization to low-energy coefficients
4. **Weighted averaging**: Supports performance-based and cluster-based task weighting

## Mathematical Foundation

### Task Vector Extraction

For each task $i$, we compute the task vector:

$$\tau_i = \theta_i - \theta_0$$

where $\theta_i$ is the fine-tuned model and $\theta_0$ is the base model.

### Mask Application

Given a binary mask $M$ for a parameter, we extract:
- **Signal (masked)**: $\tau_i^s = \tau_i[M]$
- **Noise (unmasked)**: $\tau_i^n = \tau_i[\neg M]$

### SVD Basis Construction

For each parameter, stack masked deltas into matrix $T \in \mathbb{R}^{D \times N}$:

$$T = [\tau_1^s, \tau_2^s, \ldots, \tau_N^s]$$

Optionally center: $T \leftarrow T - \mathbb{E}[T]$

Compute SVD: $T = U \Sigma V^T$

Select rank $k$ based on energy threshold:

$$k = \min\left\{j : \frac{\sum_{i=1}^j \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \geq \alpha, k \leq k_{max}\right\}$$

Split basis:
- $U_{high} = U[:, :k]$ (high energy)
- $U_{low} = U[:, k:]$ (low energy)

### Coefficient Projection

For each task $i$:

$$c_i^{high} = U_{high}^T \tau_i^s$$
$$c_i^{low} = U_{low}^T \tau_i^s$$

### Quantization

- Store $c_i^{high}$ in **FP16**
- Quantize $c_i^{low}$ with **multi-stage RTVQ**:
  - Stage 1: $q_1, s_1, z_1 = \text{Quantize}(c_i^{low}, b)$
  - Stage 2: $q_2, s_2, z_2 = \text{Quantize}(c_i^{low} - \text{Dequantize}(q_1, s_1, z_1), b)$
  - ...

### Weighted Merging

Compute task weights $w_i$ based on strategy:
- **Uniform**: $w_i = 1/N$
- **Performance**: $w_i \propto \exp(\text{acc}_i / T)$
- **Cluster**: Group tasks, weight within/across clusters

Average coefficients:

$$\bar{c}^{high} = \sum_{i=1}^N w_i c_i^{high}$$
$$\bar{c}^{low} = \sum_{i=1}^N w_i \text{Dequantize}(c_i^{low})$$

### Reconstruction

Reconstruct merged delta:

$$\bar{\tau}^s = U_{high} \bar{c}^{high} + U_{low} \bar{c}^{low}$$

If noise region included:

$$\bar{\tau}^n = U_{high}^n \bar{c}^{high,n} + U_{low}^n \bar{c}^{low,n}$$

Apply shrinkage: $\bar{\tau}^n \leftarrow \lambda \bar{\tau}^n$

Reconstruct full parameter: $\bar{\tau} = M \odot \bar{\tau}^s + (\neg M) \odot \bar{\tau}^n$

Apply to base: $\bar{\theta} = \theta_0 + \bar{\tau}$

## Usage

### Full Example: CLIP ViT-B/32 with 8 Tasks

Complete example merging 8 task-specific CLIP models with performance weighting:

```bash
python src/main.py --method svd_hybrid --model ViT-B-32 \
  --energy-threshold 0.95 --max-rank 64 \
  --tasks Eurosat Cars DTD SUN397 RESISC45 SVHN GTSRB MNIST \
  --low-bits 4 --rtvq-stages 2 --weighting performance \
  --mask-strategy union --store-artifacts --eval-reconstruction \
  --checkpoint-dir ./checkpoints \
  --base-model-path ./base_model.pt \
  --mask-dir ./masks \
  --output-dir ./output
```

This will produce:
- `output/merged_state_dict.pt` - Merged model
- `output/diagnostics.json` - Detailed metrics
- `output/weights.json` - Task weights used
- `artifacts/` - Bases, coefficients, and quantization payloads

### Basic Command

```bash
python src/main.py --method svd_hybrid \
  --tasks cars eurosat dtd sun397 \
  --checkpoint-dir ./checkpoints \
  --base-model-path ./base_model.pt \
  --mask-dir ./masks \
  --output-dir ./output
```

### Advanced Options

```bash
python scripts/run_svd_hybrid.py \
  --tasks cars eurosat dtd sun397 \
  --checkpoint-dir ./checkpoints \
  --base-model-path ./base_model.pt \
  --mask-dir ./masks \
  --energy-threshold 0.95 \
  --max-rank 64 \
  --low-bits 4 \
  --rtvq-stages 3 \
  --mask-strategy union \
  --weighting performance \
  --performance-file ./accuracies.json \
  --store-artifacts \
  --output-dir ./output \
  --artifact-dir ./artifacts
```

### Clustering Mode

```bash
python scripts/run_svd_hybrid.py \
  --tasks task1 task2 task3 task4 task5 task6 \
  --checkpoint-dir ./checkpoints \
  --base-model-path ./base_model.pt \
  --weighting cluster \
  --cluster-k 3 \
  --output-dir ./output
```

### Hydra Configuration

Create `my_config.yaml`:

```yaml
method: svd_hybrid
tasks: [cars, eurosat, dtd, sun397]
checkpoint_dir: ./checkpoints
base_model_path: ./base_model.pt
mask_dir: ./masks
output_dir: ./output

method:
  svd_energy_threshold: 0.90
  svd_max_rank: 128
  svd_weighting: performance
  performance_file: ./accuracies.json
```

Run:

```bash
python src/svd_hybrid/hydra_entry.py --config-name my_config
```

## Performance File Format

JSON file mapping task names to accuracy values:

```json
{
  "cars": 0.85,
  "eurosat": 0.92,
  "dtd": 0.78,
  "sun397": 0.81
}
```

## Artifact Storage

When `--store-artifacts` is enabled, the following are saved:

```
artifacts/
├── basis/
│   ├── layer1.weight.pt
│   ├── layer2.weight.pt
│   └── ...
├── coeffs/
│   ├── layer1.weight.pt
│   ├── layer2.weight.pt
│   └── ...
├── diagnostics.json
└── config.json
```

### Reconstructing from Artifacts

Reconstruct the merged model from saved artifacts without needing the original task checkpoints:

```bash
# Using the reload module
python -m src.svd_hybrid.reload \
  --artifact-dir ./artifacts \
  --base-model-path ./base_model.pt \
  --output-path ./reconstructed_model.pt

# Or using the root-level script
python load_and_merge.py \
  --artifact-dir ./artifacts \
  --base-model-path ./base_model.pt \
  --output-path ./reconstructed_model.pt
```

## Diagnostics

The `diagnostics.json` file contains:

- **Per-parameter metrics**:
  - Selected rank $k$
  - Energy retained
  - Reconstruction errors per task
  - Compression ratios
  
- **Summary statistics**:
  - Average rank across parameters
  - Average energy retained
  - Average reconstruction error
  - Overall compression ratio

Example output:

```json
{
  "summary": {
    "num_parameters": 150,
    "average_rank": 42.3,
    "average_energy_retained": 0.925,
    "average_reconstruction_error": 0.0023,
    "average_compression_ratio": 12.4
  },
  "per_parameter": {
    "encoder.layer.0.attention.self.query.weight": {
      "k": 48,
      "energy_retained": 0.932,
      "mean_relative_error": 0.0019,
      "compression_ratios": {
        "task1": 15.2,
        "task2": 14.8
      }
    }
  }
}
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `svd_energy_threshold` | 0.90 | Fraction of energy to retain |
| `svd_max_rank` | 128 | Maximum rank cap |
| `svd_center` | true | Center task matrix |
| `svd_fp16` | true | Use FP16 for bases |
| `svd_low_bits` | 4 | Quantization bits |
| `svd_rtvq_stages` | 2 | RTVQ refinement stages |
| `svd_mask_strategy` | union | Mask combination (union/intersection/majority) |
| `svd_include_noise` | false | Process unmasked region |
| `svd_noise_shrink` | 0.5 | Noise shrinkage factor |
| `svd_weighting` | uniform | Weighting strategy |
| `svd_cluster_k` | 2 | Number of clusters |

## Best Practices

1. **Energy threshold**: Start with 0.90-0.95 for good compression vs quality trade-off
2. **Quantization**: Use 4 bits with 2-3 stages for most cases
3. **Weighting**: Use performance-based when validation accuracies are available
4. **Clustering**: Useful when tasks naturally group into domains (5+ tasks)
5. **Noise region**: Only include if masks are highly selective (<30% parameters)

## Troubleshooting

### SVD Fails
- Try CPU device: `--device cpu`
- Reduce max_rank: `--max-rank 64`
- Check for NaN/Inf in task vectors

### High Reconstruction Error
- Increase energy threshold: `--energy-threshold 0.95`
- Increase quantization bits: `--low-bits 6`
- Add more RTVQ stages: `--rtvq-stages 3`

### Out of Memory
- Reduce max_rank: `--max-rank 32`
- Use FP16: `--fp16`
- Process on CPU: `--device cpu`

## References

- Task Arithmetic: [Paper](https://arxiv.org/abs/2212.04089)
- Tall Masks: [Paper](https://arxiv.org/abs/2401.01894)
- TVQ (Task Vector Quantization): Related work on compression
