"""
High-level orchestrator for SVD-Hybrid merging pipeline.

=== TUTORIAL: The Run Module ===

This module serves as the main entry point when called from the dispatcher (src/main.py).
It provides a clean interface for programmatic use of SVD-Hybrid.

=== WHEN TO USE ===

Use this module when you want to:
- Run SVD-Hybrid from Python code
- Integrate SVD-Hybrid into a larger pipeline
- Use the dispatcher with --method svd_hybrid

=== EXAMPLE ===

    >>> from src.svd_hybrid import SVDHybridConfig, run_svd_hybrid
    >>> 
    >>> config = SVDHybridConfig(
    ...     tasks=["Cars", "DTD", "EuroSAT", "SUN397"],
    ...     checkpoint_dir="./checkpoints",
    ...     base_model_path="./base.pt",
    ...     svd_energy_threshold=0.95,
    ...     svd_max_rank=64
    ... )
    >>> 
    >>> results = run_svd_hybrid(config)
    >>> 
    >>> # Use the merged model
    >>> merged_model = results["merged_state_dict"]
    >>> torch.save(merged_model, "merged.pt")
"""

from typing import Dict
from .config import SVDHybridConfig
from .cli import run_svd_hybrid_pipeline


def run_svd_hybrid(config: SVDHybridConfig) -> Dict:
    """
    Run the SVD-Hybrid merging pipeline.
    
    This is the main function called by the dispatcher. It delegates to
    the run_svd_hybrid_pipeline function in cli.py.
    
    === WHAT IT DOES ===
    
    1. Validates configuration
    2. Loads task vectors and masks
    3. Constructs SVD bases
    4. Compresses and merges
    5. Saves results
    
    Args:
        config: SVDHybridConfig object with all parameters
        
    Returns:
        Dictionary containing:
            - merged_state_dict: The merged model
            - diagnostics: Quality metrics
            - bases: SVD bases (if stored)
            - compressed: Compressed data (if stored)
    """
    return run_svd_hybrid_pipeline(config)


def main(args=None):
    """
    Main entry point when called from src/main.py dispatcher.
    
    This function is called when using:
        python src/main.py --method svd_hybrid [options]
    
    Args:
        args: Parsed command-line arguments from dispatcher (unused, kept for compatibility)
        
    Returns:
        Results dictionary from run_svd_hybrid_pipeline
    """
    # The dispatcher passes args but cli_main handles its own argument parsing,
    # so we simply delegate. The sys.argv has already been reconstructed by the dispatcher.
    from .cli import main as cli_main
    
    return cli_main()


if __name__ == "__main__":
    # If run directly, use CLI
    from .cli import main as cli_main
    cli_main()
