# SVD-Quantization-Task-Merging

Advanced model merging techniques combining SVD-based compression with task vector quantization.

## Overview

This repository implements multiple merging methods for multi-task models. It provides tools for:

1. **Task Vector Quantization (TVQ)**: Compress task vectors (model deltas) for efficient storage
2. **SVD-Hybrid Merging**: Advanced merging combining SVD decomposition with quantization
3. **Multi-Task Model Merging**: Combine multiple fine-tuned models into a single model

### What are Task Vectors?

A task vector represents the difference between a fine-tuned model and its base model:
```
task_vector = finetuned_model_weights - base_model_weights
```

By storing and manipulating task vectors instead of full model weights, we can:
- Efficiently combine multiple fine-tuned models
- Reduce storage requirements through quantization
- Enable flexible model merging strategies

### SVD-Hybrid (Tall Mask + TVQ)

An advanced merging method that combines:
- **Tall Mask localization**: Binary masks to identify task-specific parameters
- **SVD basis construction**: Energy-based decomposition of task vectors
- **Residual quantization**: Multi-stage RTVQ for efficient storage
- **Weighted averaging**: Performance-based and cluster-based task weighting

Key features:
- FP16 storage for high-energy coefficients
- 4-bit multi-stage residual quantization for low-energy coefficients
- Per-parameter SVD with energy-based rank selection
- Support for masked (signal) and unmasked (noise) regions
- Full artifact storage and reload capability
- Comprehensive diagnostics and compression metrics

## Quick Start

### Installation

```bash
# Required dependencies
pip install torch numpy scikit-learn scipy

# Optional: Hydra configuration framework support
pip install hydra-core

# Optional: For development and testing
pip install pytest
```

### Running SVD-Hybrid

```bash
python src/main.py --method svd_hybrid \
  --tasks cars eurosat dtd sun397 \
  --checkpoint-dir ./checkpoints \
  --base-model-path ./base_model.pt \
  --mask-dir ./masks \
  --output-dir ./output
```

See [docs/svd_hybrid.md](docs/svd_hybrid.md) for detailed documentation.

## Repository Structure

```
.
├── src/                          # Main source code
│   ├── __init__.py              # Source package init
│   ├── main.py                  # Main entry point / method dispatcher
│   └── svd_hybrid/              # SVD-Hybrid implementation
│       ├── __init__.py          # Package exports
│       ├── config.py            # Configuration dataclass
│       ├── mask_loader.py       # Tall Masks loading and combination
│       ├── task_vector_loader.py # Task vector extraction from checkpoints
│       ├── basis.py             # SVD basis construction with energy-based rank selection
│       ├── rtvq.py              # Residual Task Vector Quantization
│       ├── weighting.py         # Task weighting strategies (uniform, performance, cluster)
│       ├── clustering.py        # Task clustering for grouped merging
│       ├── compress.py          # Compression pipeline (project + quantize)
│       ├── merge.py             # Merging logic with weighted averaging
│       ├── storage.py           # Artifact storage and loading
│       ├── diagnostics.py       # Metrics, error analysis, and reporting
│       ├── cli.py               # Command-line interface
│       ├── hydra_entry.py       # Hydra configuration framework integration
│       ├── reload.py            # Reload merged models from artifacts
│       └── run.py               # High-level pipeline orchestrator
│
├── scripts/                      # Convenience scripts
│   ├── run_svd_hybrid.py        # Run SVD-Hybrid from command line
│   └── reload_svd_hybrid.py     # Reload and verify merged models
│
├── config/                       # Hydra YAML configuration files
│   │                            # (Used with hydra_entry.py for declarative config)
│   ├── config.yaml              # Main Hydra config with defaults
│   └── method/
│       └── svd_hybrid.yaml      # SVD-Hybrid specific parameters
│
├── configs/                      # JSON configuration files
│   │                            # (Used with CLI --config options)
│   ├── load_config.json         # Config for loading quantized task vectors
│   └── quantize_config.json     # Config for quantizing task vectors
│
├── examples/                     # Usage examples
│   └── example_usage.py         # Programmatic usage examples
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_quantization_utils.py  # Quantization function tests
│   ├── test_task_vectors.py     # Task vector class tests
│   ├── test_rank_selection.py   # SVD rank selection tests
│   ├── test_rtvq.py             # RTVQ quantization tests
│   ├── test_mask_strategies.py  # Mask combination tests
│   └── test_integration.py      # End-to-end integration tests
│
├── docs/                         # Documentation
│   └── svd_hybrid.md            # Detailed SVD-Hybrid documentation
│
├── quantization_utils.py        # Core quantization functions (absmax, asymmetric)
├── task_vectors.py              # Task vector classes (TaskVector, QuantizedTaskVector, etc.)
├── dataset_constants.py         # Standard 8-task evaluation definitions
├── load_and_merge.py            # Root-level artifact reconstruction script
│
├── __init__.py                  # Root package init (legacy compatibility)
├── README.md                    # This file
├── CHANGELOG.md                 # Version history and changes
└── IMPLEMENTATION_SUMMARY.md    # Technical implementation details
```

### Configuration Directories Explained

This repository has **two configuration directories** serving different purposes:

#### `config/` - Hydra YAML Configurations
Used with the [Hydra](https://hydra.cc/) configuration framework for declarative, composable configs:
- `config.yaml`: Main configuration with defaults and method selection
- `method/svd_hybrid.yaml`: Method-specific parameters

Use with:
```bash
python src/svd_hybrid/hydra_entry.py --config-name config
```

#### `configs/` - JSON Configuration Files
Used with command-line `--config` options for JSON-based configuration:
- `load_config.json`: Settings for loading pre-quantized task vectors
- `quantize_config.json`: Settings for quantizing task vectors

Use with:
```bash
python src/main.py --method svd_hybrid --load-config configs/load_config.json
```

## Features

### SVD-Hybrid Method

- **Mask Strategies**: Union, intersection, or majority voting for combining tall masks
- **Adaptive Rank Selection**: Energy-based threshold with configurable caps
- **Multi-Stage Quantization**: 2-8 bit RTVQ with residual refinement
- **Task Weighting**: 
  - Uniform (equal weights)
  - Performance-based (softmax of validation accuracies)
  - Cluster-based (group similar tasks)
- **Artifact Storage**: Save bases, coefficients, and diagnostics for later reconstruction
- **Comprehensive Diagnostics**: Per-parameter and global reconstruction errors, compression ratios

### Example Configurations

**Full Example (CLIP ViT-B/32 with 8 Tasks)**:
```bash
python src/main.py --method svd_hybrid --model ViT-B-32 \
  --energy-threshold 0.95 --max-rank 64 \
  --tasks Eurosat Cars DTD SUN397 RESISC45 SVHN GTSRB MNIST \
  --low-bits 4 --rtvq-stages 2 --weighting performance \
  --mask-strategy union --store-artifacts --eval-reconstruction
```

**Basic Merge (Uniform Weighting)**:
```bash
python src/main.py --method svd_hybrid \
  --tasks task1 task2 task3 task4 \
  --checkpoint-dir ./ckpts \
  --base-model-path ./base.pt \
  --output-dir ./output
```

**Performance-Weighted Merge**:
```bash
python scripts/run_svd_hybrid.py \
  --tasks task1 task2 task3 task4 \
  --checkpoint-dir ./ckpts \
  --base-model-path ./base.pt \
  --weighting performance \
  --performance-file ./accuracies.json \
  --output-dir ./output
```

**Cluster-Based Merge**:
```bash
python scripts/run_svd_hybrid.py \
  --tasks task1 task2 task3 task4 task5 task6 \
  --checkpoint-dir ./ckpts \
  --base-model-path ./base.pt \
  --weighting cluster \
  --cluster-k 3 \
  --output-dir ./output
```

**High-Quality Merge with Artifacts**:
```bash
python scripts/run_svd_hybrid.py \
  --tasks task1 task2 task3 task4 \
  --checkpoint-dir ./ckpts \
  --base-model-path ./base.pt \
  --mask-dir ./masks \
  --energy-threshold 0.95 \
  --max-rank 64 \
  --low-bits 6 \
  --rtvq-stages 3 \
  --store-artifacts \
  --artifact-dir ./artifacts \
  --output-dir ./output
```

### Artifact Reconstruction and Reload

After running SVD-Hybrid with `--store-artifacts`, you can reload the merged model from saved artifacts without needing the original finetuned checkpoints:

```bash
# Reload and verify
python scripts/reload_svd_hybrid.py \
  --artifact-dir ./svd_hybrid_output/ViT-B-32/artifacts \
  --verify \
  --merged-model-path ./svd_hybrid_output/merged_state_dict.pt \
  --output-path ./reloaded_merged.pt

# Or just reload without verification
python scripts/reload_svd_hybrid.py \
  --artifact-dir ./svd_hybrid_output/ViT-B-32/artifacts \
  --output-path ./reloaded_merged.pt
```

The artifacts directory contains:
- `bases/` - SVD bases for each parameter (U_high, U_low, singular values)
- `compressed/` - Quantized coefficients for each task
- `diagnostics.json` - Reconstruction errors and compression metrics
- `config.json` - Configuration used for merging
- `weights.json` - Task weights used
- `clusters.json` - Cluster assignments (if clustering was used)

## Core Modules

### Task Vector Quantization (TVQ)

The repository includes core TVQ utilities:

**quantization_utils.py** - Core quantization functions:
- `absmax_quantization(X, qbit)` - Symmetric signed quantization (int8/int16)
- `asymmetric_quantization(X, qbit)` - Min-max unsigned quantization (uint8)
- `quantization_error_check()` - Compute L1/L2 reconstruction errors
- `quantization_error_check_asymmetric()` - Error metrics for asymmetric quantization

**task_vectors.py** - Task vector classes:
- `TaskVector` - Compute and manipulate task vectors (delta = finetuned - pretrained)
- `QuantizedTaskVector` - Reconstruct from quantized deltas
- `QuantizedFinetunedModel` - Store finetuned model as quantized weights
- `QuantizedBaseAndTaskVector` - Residual storage (quantized base + quantized delta)

Example usage:
```python
from task_vectors import TaskVector
from quantization_utils import asymmetric_quantization, quantization_error_check_asymmetric

# Create task vector
tv = TaskVector("base_model.pt", "finetuned_model.pt", task_name="Cars")

# Apply to base model
merged = tv.apply_to("base_model.pt")

# Quantize
for key, delta in tv.vector.items():
    q_indices, scale, zero_point = asymmetric_quantization(delta, qbit=8)
    metrics = quantization_error_check_asymmetric(delta, q_indices, scale, zero_point)
    print(f"{key}: L1 relative error = {metrics['l1_relative']:.6f}")
```

### Dataset Constants

The 8 standard evaluation tasks are centralized in `dataset_constants.py`:
```python
from dataset_constants import STANDARD_8_TASKS, normalize_task_name

tasks = STANDARD_8_TASKS  # ['EuroSAT', 'Cars', 'DTD', 'SUN397', 'RESISC45', 'SVHN', 'GTSRB', 'MNIST']
```

## Testing

Run tests to verify implementation:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_rank_selection.py -v
pytest tests/test_rtvq.py -v
pytest tests/test_mask_strategies.py -v
```

## Documentation

- [docs/svd_hybrid.md](docs/svd_hybrid.md) - Detailed SVD-Hybrid documentation with mathematical foundations, usage examples, and troubleshooting

## Citation

If you use this work, please cite the relevant papers:
- Task Arithmetic
- Tall Masks
- Task Vector Quantization (TVQ)
