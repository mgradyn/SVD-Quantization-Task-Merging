"""
SVD-Hybrid merging method combining Tall Mask localization with TVQ quantization.

=== TUTORIAL: What is SVD-Hybrid? ===

SVD-Hybrid is an advanced model merging technique that efficiently combines
multiple fine-tuned models into a single multi-task model. It uses:

1. **SVD (Singular Value Decomposition)**: Finds the most important directions
   in the task vector space and separates high-energy from low-energy components.

2. **Tall Masks**: Binary masks that identify which parameters are important
   for each task. Only the "signal" (masked) portions are prioritized.

3. **Quantization (RTVQ)**: Low-energy components are quantized to 4-bit
   integers for compression while high-energy components remain in FP16.

4. **Weighted Merging**: Tasks can be weighted uniformly, by performance,
   or by cluster similarity.

=== WHY USE SVD-HYBRID? ===

- **Better compression**: ~12x compression ratio vs. storing full models
- **Quality preservation**: Energy-based rank selection retains important features
- **Flexibility**: Supports multiple weighting and masking strategies
- **Reproducibility**: Artifacts can be saved and used to reconstruct the merge

=== QUICK START ===

    >>> from src.svd_hybrid import SVDHybridConfig, run_svd_hybrid
    >>> 
    >>> # Create configuration
    >>> config = SVDHybridConfig(
    ...     tasks=["Cars", "DTD", "EuroSAT", "SUN397"],
    ...     checkpoint_dir="./checkpoints",
    ...     base_model_path="./base_model.pt",
    ...     svd_energy_threshold=0.95,
    ...     svd_max_rank=64,
    ...     output_dir="./output"
    ... )
    >>> 
    >>> # Run the merging pipeline
    >>> results = run_svd_hybrid(config)
    >>> 
    >>> # Access the merged model
    >>> merged_model = results["merged_state_dict"]

=== COMMAND LINE USAGE ===

    python src/main.py --method svd_hybrid \\
        --tasks Cars DTD EuroSAT SUN397 \\
        --checkpoint-dir ./checkpoints \\
        --base-model-path ./base_model.pt \\
        --energy-threshold 0.95 \\
        --max-rank 64 \\
        --output-dir ./output
"""

__version__ = "0.1.0"

# Import main interfaces for convenient access
from .config import SVDHybridConfig
from .run import run_svd_hybrid

# Define what gets exported with "from svd_hybrid import *"
__all__ = ["SVDHybridConfig", "run_svd_hybrid"]
