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

### Google Colab Notebook

For running experiments with CLIP models (ViT-B-16, ViT-B-32, ViT-L-14) on 8 tasks, use the provided Colab notebook:

[![Open In Colab](https://colab.research.google.com/drive/1siA6gTAHgTGk_NpsxyKsoNUsEpoqOJ9w?usp=sharing)]

The notebook expects checkpoints in your Google Drive with this structure:
```
My Drive/clip_finetune/
├── ViT-B-16/
│   ├── Cars/finetuned.pt
│   ├── DTD/finetuned.pt
│   ├── EuroSAT/finetuned.pt
│   ├── GTSRB/finetuned.pt
│   ├── MNIST/finetuned.pt
│   ├── RESISC45/finetuned.pt
│   ├── SUN397/finetuned.pt
│   ├── SVHN/finetuned.pt
│   └── {task}_head.pt (classification heads)
├── ViT-B-32/
│   └── ... (same structure)
└── ViT-L-14/
    └── ... (same structure)
```

## Repository Structure

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
