#!/usr/bin/env python3
"""
Example usage of SVD-Hybrid merging method.

=== TUTORIAL: SVD-Hybrid Examples ===

This script demonstrates how to use the SVD-Hybrid method programmatically.
Each example shows a different configuration pattern.

=== RUNNING EXAMPLES ===

These examples are meant to be adapted to your use case. They use placeholder
paths that you'll need to replace with your actual checkpoint locations.

=== EXAMPLE PATTERNS ===

1. **Basic Merge**: Simplest configuration with uniform weighting
2. **Performance-Weighted**: Weight tasks by their validation accuracy  
3. **Cluster-Based**: Group similar tasks before merging
4. **High-Quality**: Maximum accuracy settings

=== TYPICAL WORKFLOW ===

1. Fine-tune models on different tasks
2. Save checkpoints to a directory
3. Configure SVD-Hybrid with your paths
4. Run the pipeline
5. Load merged model into your application
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.svd_hybrid.config import SVDHybridConfig
from src.svd_hybrid.cli import run_svd_hybrid_pipeline


def example_basic_merge():
    """
    Example 1: Basic Merge with Uniform Weighting
    
    The simplest configuration - all tasks get equal weight.
    Good starting point for most use cases.
    """
    print("=" * 60)
    print("Example 1: Basic Merge with Uniform Weighting")
    print("=" * 60)
    
    # Create configuration with minimal required settings
    config = SVDHybridConfig(
        # Required: List of task names (should match checkpoint filenames)
        tasks=["cars", "eurosat", "dtd", "sun397"],
        
        # Required: Directory containing task checkpoints
        # Expected files: cars.pt, eurosat.pt, dtd.pt, sun397.pt
        checkpoint_dir="./checkpoints",
        
        # Required: Path to base (pretrained) model
        base_model_path="./base_model.pt",
        
        # Optional: Directory with Tall Mask files (empty = no masks)
        mask_dir="",
        
        # SVD Parameters
        svd_energy_threshold=0.90,  # Retain 90% of energy
        svd_max_rank=128,           # Cap rank at 128
        
        # Weighting: All tasks get equal weight
        svd_weighting="uniform",
        
        # Output
        output_dir="./output_basic",
        device="cuda"  # Use "cpu" if no GPU
    )
    
    # Note: This would require actual checkpoint files to run
    # Uncomment to run: results = run_svd_hybrid_pipeline(config)
    
    print("\nConfiguration:")
    print(f"  Tasks: {config.tasks}")
    print(f"  Energy threshold: {config.svd_energy_threshold}")
    print(f"  Max rank: {config.svd_max_rank}")
    print(f"  Weighting: {config.svd_weighting}")
    print("\nTo run: Provide valid checkpoint paths and execute")


def example_performance_weighted():
    """
    Example 2: Performance-Weighted Merge
    
    Weight tasks by their validation accuracy.
    Better-performing tasks have more influence on the merge.
    """
    print("\n" + "=" * 60)
    print("Example 2: Performance-Weighted Merge")
    print("=" * 60)
    
    config = SVDHybridConfig(
        tasks=["cars", "eurosat", "dtd", "sun397"],
        checkpoint_dir="./checkpoints",
        base_model_path="./base_model.pt",
        
        # Use performance-based weighting
        svd_weighting="performance",
        
        # Path to JSON file with task accuracies
        # Format: {"cars": 0.85, "eurosat": 0.92, "dtd": 0.73, "sun397": 0.68}
        performance_file="./task_accuracies.json",
        
        # Temperature controls weight sharpness
        # Higher = more uniform, Lower = favor best tasks
        svd_weighting_temperature=5.0,
        
        output_dir="./output_performance"
    )
    
    print("\nConfiguration:")
    print(f"  Weighting: {config.svd_weighting}")
    print(f"  Performance file: {config.performance_file}")
    print(f"  Temperature: {config.svd_weighting_temperature}")


def example_cluster_based():
    """
    Example 3: Cluster-Based Merge
    
    Group similar tasks into clusters, then merge.
    Good when some tasks are more related than others.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cluster-Based Merge")
    print("=" * 60)
    
    config = SVDHybridConfig(
        tasks=["cars", "eurosat", "dtd", "sun397", "mnist", "svhn"],
        checkpoint_dir="./checkpoints",
        base_model_path="./base_model.pt",
        
        # Use cluster-based weighting
        svd_weighting="cluster",
        
        # Number of clusters (e.g., 3 groups of similar tasks)
        svd_cluster_k=3,
        
        output_dir="./output_cluster"
    )
    
    print("\nConfiguration:")
    print(f"  Tasks: {config.tasks}")
    print(f"  Weighting: {config.svd_weighting}")
    print(f"  Number of clusters: {config.svd_cluster_k}")


def example_high_quality():
    """
    Example 4: High-Quality Merge with Artifacts
    
    Maximum quality settings with full artifact storage.
    Use when accuracy is more important than compression.
    """
    print("\n" + "=" * 60)
    print("Example 4: High-Quality Merge")
    print("=" * 60)
    
    config = SVDHybridConfig(
        tasks=["cars", "eurosat", "dtd", "sun397"],
        checkpoint_dir="./checkpoints",
        base_model_path="./base_model.pt",
        mask_dir="./masks",  # Use Tall Masks
        
        # High energy retention for accuracy
        svd_energy_threshold=0.99,
        svd_max_rank=256,
        
        # More bits for better quantization
        svd_low_bits=6,
        svd_rtvq_stages=3,
        
        # Include noise region processing
        svd_include_noise=True,
        svd_noise_shrink=0.3,
        
        # Store everything for later analysis
        svd_store_artifacts=True,
        svd_eval_reconstruction=True,
        
        output_dir="./output_hq",
        artifact_dir="./artifacts_hq"
    )
    
    print("\nConfiguration:")
    print(f"  Energy threshold: {config.svd_energy_threshold}")
    print(f"  Max rank: {config.svd_max_rank}")
    print(f"  Low bits: {config.svd_low_bits}")
    print(f"  RTVQ stages: {config.svd_rtvq_stages}")
    print(f"  Store artifacts: {config.svd_store_artifacts}")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("SVD-Hybrid Merging Method - Usage Examples")
    print("*" * 60)
    print("\nThese examples demonstrate various usage patterns.")
    print("Replace placeholder paths with your actual checkpoint locations.")
    
    # Run all examples
    example_basic_merge()
    example_performance_weighted()
    example_cluster_based()
    example_high_quality()
    
    print("\n" + "=" * 60)
    print("For more details, see docs/svd_hybrid.md")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
