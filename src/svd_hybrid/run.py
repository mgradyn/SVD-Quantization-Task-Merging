"""
High-level orchestrator for SVD-Hybrid merging pipeline.

This module serves as the main entry point when called from the dispatcher (src/main.py).
It wraps the CLI pipeline functionality for programmatic use.
"""
from typing import Dict
from .config import SVDHybridConfig
from .cli import run_svd_hybrid_pipeline


def run_svd_hybrid(config: SVDHybridConfig) -> Dict:
    """
    Run the SVD-Hybrid merging pipeline.
    
    This is the main function called by the dispatcher. It delegates to
    the run_svd_hybrid_pipeline function in cli.py.
    
    Args:
        config: SVDHybridConfig object with all parameters
        
    Returns:
        Dictionary containing merged_state_dict, diagnostics, bases, and compressed data
    """
    return run_svd_hybrid_pipeline(config)


def main(args):
    """
    Main entry point when called from src/main.py dispatcher.
    
    Args:
        args: Parsed command-line arguments from dispatcher
        
    Returns:
        Results dictionary from run_svd_hybrid_pipeline
    """
    # Import here to avoid circular imports
    from .cli import main as cli_main
    
    # Delegate to CLI main which handles argument parsing and execution
    return cli_main()


if __name__ == "__main__":
    # If run directly, use CLI
    from .cli import main as cli_main
    cli_main()
