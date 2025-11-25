"""
Configuration dataclass for SVD-Hybrid merging method.

This module defines the SVDHybridConfig dataclass that holds all configuration
parameters for the SVD-Hybrid merging pipeline. Using a dataclass provides:

- Type checking and documentation for all parameters
- Default values that work for common use cases
- Automatic validation of parameter values
- Easy serialization to/from JSON

=== TUTORIAL: Configuration Parameters ===

The configuration is organized into several categories:

1. **SVD Parameters**: Control how task vectors are decomposed
   - svd_energy_threshold: How much variance to retain (0.9 = 90%)
   - svd_max_rank: Maximum number of SVD components to keep
   - svd_center: Whether to mean-center before SVD
   - svd_fp16: Use FP16 for bases (memory efficient)

2. **Quantization Parameters**: Control compression of low-energy components
   - svd_low_bits: Bit width for quantization (4 = 16 levels)
   - svd_rtvq_stages: Number of residual refinement stages

3. **Mask Parameters**: Control how Tall Masks are combined
   - svd_mask_strategy: union (OR), intersection (AND), or majority voting
   - svd_include_noise: Process unmasked regions too

4. **Weighting Parameters**: Control how tasks are weighted in the merge
   - svd_weighting: uniform, performance-based, or cluster-based
   - svd_weighting_temperature: Softmax temperature for performance weighting
   - svd_cluster_k: Number of clusters for cluster-based weighting

5. **Storage Parameters**: Control what gets saved
   - svd_store_artifacts: Save bases and coefficients for later reconstruction
   - svd_eval_reconstruction: Compute reconstruction error metrics

=== EXAMPLE CONFIGURATIONS ===

Basic (default settings):
    >>> config = SVDHybridConfig(
    ...     tasks=["Cars", "DTD"],
    ...     checkpoint_dir="./ckpts",
    ...     base_model_path="./base.pt"
    ... )

High quality (more rank, more bits):
    >>> config = SVDHybridConfig(
    ...     tasks=["Cars", "DTD"],
    ...     checkpoint_dir="./ckpts",
    ...     base_model_path="./base.pt",
    ...     svd_energy_threshold=0.99,
    ...     svd_max_rank=128,
    ...     svd_low_bits=6,
    ...     svd_rtvq_stages=3
    ... )

Performance-weighted:
    >>> config = SVDHybridConfig(
    ...     tasks=["Cars", "DTD"],
    ...     checkpoint_dir="./ckpts",
    ...     base_model_path="./base.pt",
    ...     svd_weighting="performance",
    ...     performance_file="./accuracies.json"
    ... )
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SVDHybridConfig:
    """
    Configuration for SVD-Hybrid merging method.
    
    All parameters have sensible defaults that work for most use cases.
    Only tasks, checkpoint_dir, and base_model_path are typically required.
    
    Attributes:
        === SVD PARAMETERS ===
        svd_energy_threshold: Fraction of cumulative energy (variance) to retain.
            Higher values = more accuracy, less compression. Range: (0, 1]
            Default: 0.95 (retain 95% of energy)
            
        svd_max_rank: Maximum rank (number of SVD components) regardless of energy.
            Caps the rank even if energy threshold isn't met.
            Default: 64 components
            
        svd_center: Whether to mean-center the task matrix before SVD.
            Centering often improves decomposition quality.
            Default: True
            
        svd_fp16: Store SVD bases (U matrices) in FP16 instead of FP32.
            Halves memory usage with minimal quality loss.
            Default: True
            
        === QUANTIZATION PARAMETERS ===
        svd_low_bits: Number of bits for quantizing low-energy coefficients.
            Lower = more compression, higher error. Range: [1, 8]
            Default: 4 bits (16 quantization levels)
            
        svd_rtvq_stages: Number of residual quantization stages.
            More stages = better approximation of low-energy components.
            Default: 2 stages
            
        === MASK PARAMETERS ===
        svd_mask_strategy: How to combine masks from different tasks.
            - "union": Include parameter if ANY task uses it (OR)
            - "intersection": Include only if ALL tasks use it (AND)
            - "majority": Include if majority of tasks use it
            Default: "union"
            
        svd_include_noise: Whether to process unmasked (noise) region.
            If True, processes both masked and unmasked regions separately.
            Default: False
            
        === WEIGHTING PARAMETERS ===
        svd_weighting: Task weighting strategy for the final merge.
            - "uniform": Equal weight for all tasks
            - "performance": Weight by task validation accuracy
            - "cluster": Group similar tasks, equal weight within groups
            Default: "uniform"
            
        svd_weighting_temperature: Temperature for performance-based softmax.
            Higher = more uniform, lower = more concentrated on best tasks.
            Default: 5.0
            
        svd_cluster_k: Number of clusters for cluster-based weighting.
            Only used when svd_weighting="cluster".
            Default: 2
            
        === STORAGE AND EVALUATION ===
        svd_store_artifacts: Save bases, coefficients, and diagnostics.
            Required if you want to reconstruct the merge later.
            Default: True
            
        svd_eval_reconstruction: Compute reconstruction error metrics.
            Useful for understanding compression quality.
            Default: True
            
        === ADVANCED OPTIONS ===
        svd_noise_shrink: Shrinkage factor applied to unmasked (noise) region.
            Reduces influence of unimportant parameters. Range: [0, 1]
            Default: 0.5
            
        svd_min_mask_size: Minimum number of masked elements to process.
            Skip parameters with fewer masked elements.
            Default: 10
            
        svd_randomized_svd_threshold: Use randomized SVD when D*N > threshold.
            Faster for large matrices but approximate.
            Default: 1,500,000 elements
    """
    
    # === SVD PARAMETERS ===
    # Controls how task vectors are decomposed into principal components
    svd_energy_threshold: float = 0.95  # Retain this fraction of cumulative energy
    svd_max_rank: int = 64              # Maximum rank cap (number of components)
    svd_center: bool = True             # Mean-center task matrix before SVD
    svd_fp16: bool = True               # Use FP16 for bases (memory efficient)
    
    # === QUANTIZATION PARAMETERS ===
    # Controls compression of low-energy coefficients
    svd_low_bits: int = 4               # Bits for low-energy coefficient quantization
    svd_rtvq_stages: int = 2            # Number of RTVQ refinement stages
    
    # === MASK PARAMETERS ===
    # Controls how Tall Masks from different tasks are combined
    svd_mask_strategy: str = "union"    # union, intersection, or majority
    svd_include_noise: bool = False     # Whether to process unmasked (noise) region
    
    # === WEIGHTING PARAMETERS ===
    # Controls how tasks are weighted in the final merge
    svd_weighting: str = "uniform"      # uniform, performance, or cluster
    svd_weighting_temperature: float = 5.0  # Temperature for performance-based weighting
    svd_cluster_k: int = 2              # Number of clusters for cluster-based weighting
    
    # === STORAGE AND EVALUATION ===
    svd_store_artifacts: bool = True    # Store compression artifacts for reconstruction
    svd_eval_reconstruction: bool = True  # Evaluate reconstruction error
    
    # === ADVANCED OPTIONS ===
    svd_noise_shrink: float = 0.5       # Shrinkage factor for noise region
    svd_min_mask_size: int = 10         # Minimum mask size to process
    svd_randomized_svd_threshold: int = 1500000  # Use randomized SVD if D*N > threshold
    
    # === TASK AND CHECKPOINT PATHS ===
    # These are typically the only required parameters
    tasks: List[str] = field(default_factory=list)  # List of task names (e.g., ["Cars", "DTD"])
    model: str = "ViT-B-32"             # Model identifier for organization
    checkpoint_dir: str = ""            # Directory containing task checkpoints
    base_model_path: str = ""           # Path to base/pretrained model checkpoint
    mask_dir: str = ""                  # Directory containing Tall Mask files (optional)
    
    # === PERFORMANCE METRICS ===
    performance_file: Optional[str] = None  # Path to JSON with task accuracies
    
    # === OUTPUT PATHS ===
    output_dir: str = "./svd_hybrid_output"  # Directory for merged model
    artifact_dir: str = "./artifacts"   # Directory for saved artifacts
    
    # === DEVICE ===
    device: str = "cuda"                # Device to use: "cuda" or "cpu"
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Called automatically by dataclass after __init__.
        Raises ValueError if any parameter is invalid.
        """
        # Validate mask strategy
        if self.svd_mask_strategy not in ["union", "intersection", "majority"]:
            raise ValueError(f"Invalid mask strategy: {self.svd_mask_strategy}. "
                           f"Must be one of: union, intersection, majority")
        
        # Validate weighting strategy
        if self.svd_weighting not in ["uniform", "performance", "cluster"]:
            raise ValueError(f"Invalid weighting: {self.svd_weighting}. "
                           f"Must be one of: uniform, performance, cluster")
        
        # Validate energy threshold
        if self.svd_energy_threshold <= 0 or self.svd_energy_threshold > 1:
            raise ValueError(f"Energy threshold must be in (0, 1], got {self.svd_energy_threshold}")
        
        # Validate quantization bits
        if self.svd_low_bits < 1 or self.svd_low_bits > 8:
            raise ValueError(f"Low bits must be in [1, 8], got {self.svd_low_bits}")
        
        # Validate RTVQ stages
        if self.svd_rtvq_stages < 1:
            raise ValueError(f"RTVQ stages must be >= 1, got {self.svd_rtvq_stages}")
