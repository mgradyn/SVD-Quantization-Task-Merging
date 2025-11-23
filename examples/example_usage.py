#!/usr/bin/env python3
"""
Example usage of SVD-Hybrid merging method.

This script demonstrates how to use the SVD-Hybrid method programmatically.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.svd_hybrid.config import SVDHybridConfig
from src.svd_hybrid.cli import run_svd_hybrid_pipeline


def example_basic_merge():
    """Basic example with uniform weighting."""
    print("=" * 60)
    print("Example 1: Basic Merge with Uniform Weighting")
    print("=" * 60)
    
    config = SVDHybridConfig(
        tasks=["cars", "eurosat", "dtd", "sun397"],
        checkpoint_dir="./checkpoints",
        base_model_path="./base_model.pt",
        mask_dir="",  # No masks
        svd_energy_threshold=0.90,
        svd_max_rank=128,
        svd_weighting="uniform",
        output_dir="./output_basic",
        device="cuda"  # or "cpu"
    )
    
    # Note: This would require actual checkpoint files
    # results = run_svd_hybrid_pipeline(config)
    
    print("\nConfiguration:")
    print(f"  Tasks: {config.tasks}")
    print(f"  Energy threshold: {config.svd_energy_threshold}")
    print(f"  Max rank: {config.svd_max_rank}")
    print(f"  Weighting: {config.svd_weighting}")
    print("\nTo run: Provide valid checkpoint paths and execute")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("SVD-Hybrid Merging Method - Usage Examples")
    print("*" * 60)
    print("\nThese examples demonstrate various usage patterns.")
    print("See full examples in the file for more patterns.")
    
    example_basic_merge()
    
    print("\n" + "=" * 60)
    print("For more details, see docs/svd_hybrid.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
