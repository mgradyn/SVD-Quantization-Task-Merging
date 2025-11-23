"""
Hydra entry point for SVD-Hybrid merging (Tall Masks style).
"""
try:
    import hydra
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

from .config import SVDHybridConfig
from .cli import run_svd_hybrid_pipeline


def config_from_hydra(cfg: DictConfig) -> SVDHybridConfig:
    """
    Convert Hydra config to SVDHybridConfig.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        SVDHybridConfig object
    """
    # Extract relevant fields from Hydra config
    method_cfg = cfg.get("method", {})
    
    return SVDHybridConfig(
        tasks=cfg.get("tasks", []),
        checkpoint_dir=cfg.get("checkpoint_dir", ""),
        base_model_path=cfg.get("base_model_path", ""),
        mask_dir=cfg.get("mask_dir", ""),
        svd_energy_threshold=method_cfg.get("svd_energy_threshold", 0.90),
        svd_max_rank=method_cfg.get("svd_max_rank", 128),
        svd_center=method_cfg.get("svd_center", True),
        svd_fp16=method_cfg.get("svd_fp16", True),
        svd_low_bits=method_cfg.get("svd_low_bits", 4),
        svd_rtvq_stages=method_cfg.get("svd_rtvq_stages", 2),
        svd_mask_strategy=method_cfg.get("svd_mask_strategy", "union"),
        svd_include_noise=method_cfg.get("svd_include_noise", False),
        svd_noise_shrink=method_cfg.get("svd_noise_shrink", 0.5),
        svd_weighting=method_cfg.get("svd_weighting", "uniform"),
        performance_file=method_cfg.get("performance_file"),
        svd_weighting_temperature=method_cfg.get("svd_weighting_temperature", 1.0),
        svd_cluster_k=method_cfg.get("svd_cluster_k", 2),
        svd_store_artifacts=method_cfg.get("svd_store_artifacts", True),
        svd_eval_reconstruction=method_cfg.get("svd_eval_reconstruction", True),
        output_dir=cfg.get("output_dir", "./svd_hybrid_output"),
        artifact_dir=cfg.get("artifact_dir", "./artifacts"),
        device=cfg.get("device", "cuda")
    )


def run_from_hydra(cfg: DictConfig):
    """
    Run SVD-Hybrid pipeline from Hydra config.
    
    Args:
        cfg: Hydra configuration
        
    Returns:
        Results dictionary
    """
    config = config_from_hydra(cfg)
    return run_svd_hybrid_pipeline(config)


if HYDRA_AVAILABLE:
    @hydra.main(version_base=None, config_path="../../config", config_name="config")
    def main(cfg: DictConfig):
        """Hydra main entry point."""
        return run_from_hydra(cfg)
else:
    def main(cfg=None):
        """Fallback when Hydra is not available."""
        print("Error: Hydra is not installed. Install with: pip install hydra-core")
        return None


if __name__ == "__main__":
    main()
