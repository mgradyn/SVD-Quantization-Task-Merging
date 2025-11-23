# Changelog

## [Unreleased]

### Added - SVD-Hybrid Merging Method

Complete implementation of SVD-Hybrid merging method combining Tall Mask localization with TVQ residual quantization.

#### Core Features
- **Mask Loading** (`src/svd_hybrid/mask_loader.py`)
  - Support for union, intersection, and majority voting strategies
  - Binary mask application and reconstruction utilities
  
- **Task Vector Loading** (`src/svd_hybrid/task_vector_loader.py`)
  - Flexible checkpoint loading with multiple format support
  - Task vector computation and organization
  
- **SVD Basis Construction** (`src/svd_hybrid/basis.py`)
  - Energy-based rank selection with configurable thresholds
  - Support for masked (signal) and unmasked (noise) regions
  - Randomized SVD option for large matrices
  
- **RTVQ Quantization** (`src/svd_hybrid/rtvq.py`)
  - Multi-stage residual quantization (2-8 bits)
  - Asymmetric quantization with scale and zero-point
  - Configurable refinement stages
  
- **Task Weighting** (`src/svd_hybrid/weighting.py`)
  - Uniform weighting (equal weights)
  - Performance-based weighting (softmax of accuracies)
  - Cluster-based weighting (group similar tasks)
  
- **Task Clustering** (`src/svd_hybrid/clustering.py`)
  - K-means and hierarchical clustering
  - Multi-basis merging within and across clusters
  
- **Compression Pipeline** (`src/svd_hybrid/compress.py`)
  - Coefficient projection onto SVD bases
  - FP16 storage for high-energy coefficients
  - RTVQ for low-energy coefficients
  
- **Merging Logic** (`src/svd_hybrid/merge.py`)
  - Weighted averaging with configurable strategies
  - Support for masked and noise regions
  - Cluster-based merging pathway
  
- **Artifact Storage** (`src/svd_hybrid/storage.py`)
  - Save/load bases, coefficients, and diagnostics
  - JSON configuration snapshots
  - Artifact reconstruction without original checkpoints
  
- **Diagnostics** (`src/svd_hybrid/diagnostics.py`)
  - Per-parameter reconstruction errors
  - Compression ratios
  - Energy retention statistics
  - Coefficient magnitude histograms

#### User Interface
- **CLI** (`src/svd_hybrid/cli.py`)
  - Comprehensive command-line interface
  - All parameters accessible via flags
  - Integration with main dispatcher (`src/main.py`)
  
- **Hydra Support** (`src/svd_hybrid/hydra_entry.py`)
  - Optional Hydra integration for configuration management
  - Compatible with Tall Masks style configs

#### Configuration
- **Config Files**
  - `config/config.yaml` - Main configuration
  - `config/method/svd_hybrid.yaml` - Method-specific parameters
  
- **Configuration Dataclass** (`src/svd_hybrid/config.py`)
  - Type-safe configuration with validation
  - Sensible defaults for all parameters

#### Testing
- **Unit Tests**
  - `tests/test_rank_selection.py` - SVD rank selection (6 tests)
  - `tests/test_rtvq.py` - Quantization methods (9 tests)
  
- **Integration Tests**
  - `tests/test_integration.py` - End-to-end pipeline (2 tests)
  - Synthetic data testing
  - Mask support verification

#### Documentation
- **Detailed Docs** (`docs/svd_hybrid.md`)
  - Mathematical foundations
  - Usage examples
  - Configuration parameters
  - Best practices
  - Troubleshooting guide
  
- **README Updates**
  - Quick start guide
  - Repository structure
  - Example commands
  
- **Examples** (`examples/example_usage.py`)
  - Programmatic usage patterns
  - Various configuration scenarios

#### Utilities
- **Artifact Loader** (`load_and_merge.py`)
  - Reconstruct merged models from artifacts
  - Standalone script for deployment
  
- **Launch Script** (`scripts/run_svd_hybrid.py`)
  - Convenience wrapper for CLI

#### Infrastructure
- `.gitignore` - Proper exclusions for artifacts, outputs, Python cache
- Package structure with `__init__.py` files

### Test Results
- 17/17 tests passing
- End-to-end pipeline verified
- Artifact loading verified
- All three weighting modes tested (uniform, performance, cluster)

### Acceptance Criteria Status
✅ Running `python src/main.py --method svd_hybrid` completes successfully
✅ Reconstruction error and compression ratio in diagnostics.json
✅ Artifact reload capability working
✅ Weighted averaging functional (all three modes)
✅ Cluster path runs when configured

## [0.1.0] - Previous

Initial repository with basic SVD-Hybrid concept for CLIP models.
