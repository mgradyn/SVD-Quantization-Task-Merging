"""
Load artifacts and reconstruct merged model without original checkpoints.
"""
import argparse
import torch
from src.svd_hybrid.storage import load_all_artifacts
from src.svd_hybrid.merge import merge_all_parameters, apply_merged_deltas
from src.svd_hybrid.weighting import compute_uniform_weights
from src.svd_hybrid.task_vector_loader import load_checkpoint


def reconstruct_from_artifacts(
    artifact_dir: str,
    base_model_path: str,
    output_path: str,
    device: str = "cpu"
):
    """
    Reconstruct merged model from stored artifacts.
    
    Args:
        artifact_dir: Directory containing artifacts
        base_model_path: Path to base model checkpoint
        output_path: Path to save reconstructed model
        device: Device to use
    """
    print(f"Loading artifacts from {artifact_dir}...")
    artifacts = load_all_artifacts(artifact_dir, device=device)
    
    bases = artifacts["bases"]
    compressed = artifacts["compressed"]
    config = artifacts["config"]
    diagnostics = artifacts["diagnostics"]
    
    print(f"Loaded {len(bases)} parameter bases")
    
    # Get task names from diagnostics
    task_names = list(diagnostics.get("task_weights", {}).keys())
    if not task_names:
        # Fallback: extract from compressed data
        first_param = next(iter(compressed.values()))
        task_names = list(first_param.keys())
    
    print(f"Found {len(task_names)} tasks: {task_names}")
    
    # Compute weights (use stored weights if available, otherwise uniform)
    if "task_weights" in diagnostics:
        weights = diagnostics["task_weights"]
        print("Using stored task weights")
    else:
        weights = compute_uniform_weights(task_names)
        print("Using uniform weights")
    
    # Get original shapes from diagnostics
    original_shapes = {}
    for param_name, param_diag in diagnostics.get("per_parameter", {}).items():
        if "original_shape" in param_diag:
            original_shapes[param_name] = torch.Size(param_diag["original_shape"])
    
    # Load masks if needed (not stored in artifacts currently)
    masks = {}
    
    # Merge parameters
    print("Merging parameters...")
    merged_deltas = merge_all_parameters(
        compressed,
        bases,
        masks,
        weights,
        original_shapes,
        config,
        device=device
    )
    
    # Load base model and apply deltas
    print(f"Loading base model from {base_model_path}...")
    base_state_dict = load_checkpoint(base_model_path, device=device)
    
    print("Applying deltas to base model...")
    merged_state_dict = apply_merged_deltas(base_state_dict, merged_deltas, device=device)
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    torch.save(merged_state_dict, output_path)
    
    print("Done!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reconstruct merged model from SVD-Hybrid artifacts"
    )
    
    parser.add_argument("--artifact-dir", type=str, required=True,
                       help="Directory containing artifacts")
    parser.add_argument("--base-model-path", type=str, required=True,
                       help="Path to base model checkpoint")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save reconstructed merged model")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    reconstruct_from_artifacts(
        args.artifact_dir,
        args.base_model_path,
        args.output_path,
        args.device
    )


if __name__ == "__main__":
    main()
