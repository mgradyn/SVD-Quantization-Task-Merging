"""
Command-line interface for SVD-Hybrid merging.
"""
import argparse
import torch
import os
from typing import Optional, Dict
from .config import SVDHybridConfig
from .task_vector_loader import load_task_vectors, get_task_checkpoint_paths, get_parameter_names
from .mask_loader import load_task_masks, combine_masks
from .basis import construct_masked_basis
from .compress import compress_all_parameters
from .merge import merge_all_parameters, apply_merged_deltas, merge_with_clustering
from .storage import save_all_artifacts, save_merged_model
from .diagnostics import compute_all_diagnostics, print_diagnostics_summary
from .weighting import compute_weights
from .clustering import cluster_tasks
from .mask_loader import apply_mask_to_tensor, get_unmasked_portion


def run_svd_hybrid_pipeline(config: SVDHybridConfig) -> Dict:
    """
    Run the complete SVD-Hybrid merging pipeline.
    
    Args:
        config: SVDHybridConfig object
        
    Returns:
        Dictionary containing merged model and diagnostics
    """
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Step 1: Load task vectors
    print("\n[Step 1/9] Loading task vectors...")
    task_checkpoint_paths = get_task_checkpoint_paths(config.checkpoint_dir, config.tasks)
    task_vectors = load_task_vectors(
        config.base_model_path,
        task_checkpoint_paths,
        device=device
    )
    print(f"Loaded {len(task_vectors)} task vectors")
    
    # Step 2: Load and combine masks
    print("\n[Step 2/9] Loading and combining masks...")
    if config.mask_dir and os.path.exists(config.mask_dir):
        task_masks = load_task_masks(config.mask_dir, config.tasks, device=device)
        combined_masks = combine_masks(task_masks, strategy=config.svd_mask_strategy, device=device)
        print(f"Combined masks using strategy: {config.svd_mask_strategy}")
    else:
        print("No mask directory provided, using all parameters")
        combined_masks = {}
    
    # Step 3: Get parameter names and shapes
    print("\n[Step 3/9] Extracting parameter information...")
    param_names = get_parameter_names(task_vectors)
    original_shapes = {}
    for task_vector in task_vectors.values():
        for param_name, delta in task_vector.items():
            if param_name not in original_shapes:
                original_shapes[param_name] = delta.shape
    print(f"Processing {len(param_names)} parameters")
    
    # Step 4: Construct bases for each parameter
    print("\n[Step 4/9] Constructing SVD bases...")
    bases = {}
    for param_name in param_names:
        # Extract masked and unmasked deltas
        mask = combined_masks.get(param_name)
        
        masked_deltas = []
        unmasked_deltas = []
        
        for task_name, task_vector in task_vectors.items():
            if param_name not in task_vector:
                continue
            
            delta = task_vector[param_name]
            
            if mask is not None and mask.shape == delta.shape:
                # Apply mask
                if mask.sum() >= config.svd_min_mask_size:
                    masked = apply_mask_to_tensor(delta, mask)
                    masked_deltas.append(masked)
                    
                    if config.svd_include_noise:
                        unmasked = get_unmasked_portion(delta, mask)
                        unmasked_deltas.append(unmasked)
            else:
                # No mask, use entire parameter
                masked_deltas.append(delta.flatten())
        
        if masked_deltas and len(masked_deltas[0]) > 0:
            basis = construct_masked_basis(
                masked_deltas,
                unmasked_deltas if config.svd_include_noise else None,
                energy_threshold=config.svd_energy_threshold,
                max_rank=config.svd_max_rank,
                center=config.svd_center,
                device=device,
                include_noise=config.svd_include_noise
            )
            
            # Convert to FP16 if requested
            if config.svd_fp16 and basis.get("masked") is not None:
                basis["masked"]["U_high"] = basis["masked"]["U_high"].half()
                basis["masked"]["U_low"] = basis["masked"]["U_low"].half()
                
                if basis.get("noise") is not None:
                    basis["noise"]["U_high"] = basis["noise"]["U_high"].half()
                    basis["noise"]["U_low"] = basis["noise"]["U_low"].half()
            
            bases[param_name] = basis
            
            if basis.get("masked"):
                print(f"  {param_name}: k={basis['masked']['k']}, "
                      f"energy={basis['masked']['energy_retained']:.4f}")
    
    # Step 5: Compress task vectors
    print("\n[Step 5/9] Compressing task vectors...")
    compressed_all = compress_all_parameters(
        task_vectors,
        combined_masks,
        bases,
        config,
        device=device
    )
    print(f"Compressed {len(compressed_all)} parameters")
    
    # Step 6: Compute task weights
    print("\n[Step 6/9] Computing task weights...")
    cluster_assignments = None
    
    if config.svd_weighting == "cluster":
        # Cluster tasks
        print(f"Clustering tasks into {config.svd_cluster_k} clusters...")
        cluster_assignments = cluster_tasks(task_vectors, config.svd_cluster_k, method="kmeans")
        
        # Print cluster assignments
        from .clustering import get_cluster_members
        clusters = get_cluster_members(cluster_assignments)
        for cluster_id, members in clusters.items():
            print(f"  Cluster {cluster_id}: {members}")
    
    weights = compute_weights(
        config.tasks,
        weighting_strategy=config.svd_weighting,
        performance_file=config.performance_file,
        temperature=config.svd_weighting_temperature,
        cluster_assignments=cluster_assignments
    )
    
    print("Task weights:")
    for task_name, weight in sorted(weights.items()):
        print(f"  {task_name}: {weight:.4f}")
    
    # Step 7: Merge parameters
    print("\n[Step 7/9] Merging parameters...")
    if config.svd_weighting == "cluster" and cluster_assignments is not None:
        merged_deltas = merge_with_clustering(
            compressed_all,
            bases,
            combined_masks,
            weights,
            cluster_assignments,
            original_shapes,
            config,
            device=device
        )
    else:
        merged_deltas = merge_all_parameters(
            compressed_all,
            bases,
            combined_masks,
            weights,
            original_shapes,
            config,
            device=device
        )
    print(f"Merged {len(merged_deltas)} parameters")
    
    # Step 8: Apply to base model
    print("\n[Step 8/9] Creating merged model...")
    from .task_vector_loader import load_checkpoint
    base_state_dict = load_checkpoint(config.base_model_path, device=device)
    merged_state_dict = apply_merged_deltas(base_state_dict, merged_deltas, device=device)
    print("Merged model created")
    
    # Step 9: Compute diagnostics
    print("\n[Step 9/9] Computing diagnostics...")
    if config.svd_eval_reconstruction:
        diagnostics = compute_all_diagnostics(
            task_vectors,
            compressed_all,
            bases,
            combined_masks,
            config,
            device=device
        )
        
        # Add weight information
        diagnostics["task_weights"] = weights
        if cluster_assignments:
            diagnostics["cluster_assignments"] = cluster_assignments
        
        print_diagnostics_summary(diagnostics)
    else:
        diagnostics = {"task_weights": weights}
    
    # Save artifacts if requested
    if config.svd_store_artifacts:
        print("\nSaving artifacts...")
        save_all_artifacts(
            bases,
            compressed_all,
            diagnostics,
            config,
            config.artifact_dir
        )
    
    # Save merged model
    print("\nSaving merged model...")
    save_merged_model(merged_state_dict, config.output_dir)
    
    print("\n" + "="*60)
    print("SVD-Hybrid merging complete!")
    print("="*60)
    
    return {
        "merged_state_dict": merged_state_dict,
        "diagnostics": diagnostics,
        "bases": bases,
        "compressed": compressed_all
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SVD-Hybrid merging method combining Tall Masks and TVQ"
    )
    
    # Task configuration
    parser.add_argument("--tasks", nargs="+", required=True,
                       help="List of task identifiers")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory containing task checkpoints")
    parser.add_argument("--base-model-path", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--mask-dir", type=str, default="",
                       help="Directory containing tall masks")
    
    # SVD parameters
    parser.add_argument("--energy-threshold", type=float, default=0.90,
                       help="Energy retention threshold for rank selection")
    parser.add_argument("--max-rank", type=int, default=128,
                       help="Maximum rank cap")
    parser.add_argument("--center", action="store_true", default=True,
                       help="Center task matrix before SVD")
    parser.add_argument("--no-center", action="store_false", dest="center",
                       help="Don't center task matrix")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 for bases")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                       help="Use FP32 for bases")
    
    # Quantization parameters
    parser.add_argument("--low-bits", type=int, default=4,
                       help="Bits for low-energy coefficient quantization")
    parser.add_argument("--rtvq-stages", type=int, default=2,
                       help="Number of RTVQ refinement stages")
    
    # Mask parameters
    parser.add_argument("--mask-strategy", type=str, default="union",
                       choices=["union", "intersection", "majority"],
                       help="Mask combination strategy")
    parser.add_argument("--include-noise", action="store_true",
                       help="Process unmasked (noise) region")
    parser.add_argument("--noise-shrink", type=float, default=0.5,
                       help="Shrinkage factor for noise region")
    
    # Weighting parameters
    parser.add_argument("--weighting", type=str, default="uniform",
                       choices=["uniform", "performance", "cluster"],
                       help="Task weighting strategy")
    parser.add_argument("--performance-file", type=str, default=None,
                       help="Path to performance metrics JSON file")
    parser.add_argument("--weighting-temperature", type=float, default=1.0,
                       help="Temperature for performance-based weighting")
    parser.add_argument("--cluster-k", type=int, default=2,
                       help="Number of clusters for cluster-based weighting")
    
    # Storage and evaluation
    parser.add_argument("--store-artifacts", action="store_true",
                       help="Store compression artifacts")
    parser.add_argument("--eval-reconstruction", action="store_true", default=True,
                       help="Evaluate reconstruction error")
    parser.add_argument("--no-eval-reconstruction", action="store_false",
                       dest="eval_reconstruction",
                       help="Skip reconstruction evaluation")
    
    # Output paths
    parser.add_argument("--output-dir", type=str, default="./svd_hybrid_output",
                       help="Output directory for merged model")
    parser.add_argument("--artifact-dir", type=str, default="./artifacts",
                       help="Directory for artifact storage")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Create config from args
    config = SVDHybridConfig(
        tasks=args.tasks,
        checkpoint_dir=args.checkpoint_dir,
        base_model_path=args.base_model_path,
        mask_dir=args.mask_dir,
        svd_energy_threshold=args.energy_threshold,
        svd_max_rank=args.max_rank,
        svd_center=args.center,
        svd_fp16=args.fp16,
        svd_low_bits=args.low_bits,
        svd_rtvq_stages=args.rtvq_stages,
        svd_mask_strategy=args.mask_strategy,
        svd_include_noise=args.include_noise,
        svd_noise_shrink=args.noise_shrink,
        svd_weighting=args.weighting,
        performance_file=args.performance_file,
        svd_weighting_temperature=args.weighting_temperature,
        svd_cluster_k=args.cluster_k,
        svd_store_artifacts=args.store_artifacts,
        svd_eval_reconstruction=args.eval_reconstruction,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        device=args.device
    )
    
    # Run pipeline
    results = run_svd_hybrid_pipeline(config)
    
    return results


if __name__ == "__main__":
    main()
