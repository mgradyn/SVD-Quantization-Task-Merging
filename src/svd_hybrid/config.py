"""
Configuration dataclass for SVD-Hybrid merging method.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SVDHybridConfig:
    """Configuration for SVD-Hybrid merging method."""
    
    # SVD parameters
    svd_energy_threshold: float = 0.95
    svd_max_rank: int = 64
    svd_center: bool = True
    svd_fp16: bool = True
    
    # Quantization parameters
    svd_low_bits: int = 4
    svd_rtvq_stages: int = 2
    
    # Mask parameters
    svd_mask_strategy: str = "union"  # union, intersection, majority
    svd_include_noise: bool = False
    
    # Weighting parameters
    svd_weighting: str = "uniform"  # uniform, performance, cluster
    svd_weighting_temperature: float = 5.0
    svd_cluster_k: int = 2
    
    # Storage and evaluation
    svd_store_artifacts: bool = True
    svd_eval_reconstruction: bool = True
    
    # Advanced options
    svd_noise_shrink: float = 0.5
    svd_min_mask_size: int = 10
    svd_randomized_svd_threshold: int = 1500000  # Use randomized SVD if D*N > threshold (svd_fallback_threshold)
    
    # Task and checkpoint paths
    tasks: List[str] = field(default_factory=list)
    model: str = "ViT-B-32"  # Model identifier (e.g., ViT-B-32)
    checkpoint_dir: str = ""
    base_model_path: str = ""
    mask_dir: str = ""
    
    # Performance metrics
    performance_file: Optional[str] = None
    
    # Output paths
    output_dir: str = "./svd_hybrid_output"
    artifact_dir: str = "./artifacts"
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.svd_mask_strategy not in ["union", "intersection", "majority"]:
            raise ValueError(f"Invalid mask strategy: {self.svd_mask_strategy}")
        
        if self.svd_weighting not in ["uniform", "performance", "cluster"]:
            raise ValueError(f"Invalid weighting: {self.svd_weighting}")
        
        if self.svd_energy_threshold <= 0 or self.svd_energy_threshold > 1:
            raise ValueError(f"Energy threshold must be in (0, 1], got {self.svd_energy_threshold}")
        
        if self.svd_low_bits < 1 or self.svd_low_bits > 8:
            raise ValueError(f"Low bits must be in [1, 8], got {self.svd_low_bits}")
        
        if self.svd_rtvq_stages < 1:
            raise ValueError(f"RTVQ stages must be >= 1, got {self.svd_rtvq_stages}")
