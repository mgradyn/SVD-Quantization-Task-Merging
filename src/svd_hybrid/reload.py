"""
Reload and reconstruct merged model from stored artifacts.

=== TUTORIAL: Reloading from Artifacts ===

After running SVD-Hybrid with --store-artifacts, you can reconstruct the
merged model later without needing the original fine-tuned checkpoints.

=== WHY RELOAD? ===

1. **Deployment**: Share artifacts instead of full checkpoints
2. **Storage**: Artifacts are much smaller than full models
3. **Reproducibility**: Exact reconstruction from saved data
4. **Verification**: Check that reconstruction matches original

=== WHAT'S IN ARTIFACTS? ===

    artifacts/
    ├── basis/          # SVD bases (U_high, U_low) per parameter
    ├── coeffs/         # Quantized coefficients per task
    ├── config.json     # Configuration used
    └── diagnostics.json # Metrics and statistics

=== HOW TO RELOAD ===

    # From command line
    python -m src.svd_hybrid.reload \\
        --artifact-dir ./artifacts \\
        --base-model-path ./base.pt \\
        --output-path ./reloaded.pt

    # From Python
    from src.svd_hybrid.reload import reconstruct_from_artifacts
    
    result = reconstruct_from_artifacts(
        artifact_dir="./artifacts",
        base_model_path="./base.pt",
        output_path="./reloaded.pt"
    )

=== VERIFICATION ===

To verify reconstruction matches the original:

    python scripts/reload_svd_hybrid.py \\
        --artifact-dir ./artifacts \\
        --verify \\
        --merged-model-path ./original_merged.pt
"""

import argparse
import torch
from typing import Dict, Optional
from .storage import load_all_artifacts
from .merge import merge_all_parameters, apply_merged_deltas
from .weighting import compute_uniform_weights
from .task_vector_loader import load_checkpoint


def reload_merged_model_from_artifacts(
    artifact_dir: str,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Reload merged model state dict from artifacts directory.
    
    This function expects the artifacts directory to contain a pre-computed
    merged model file (merged_state_dict.pt) or will reconstruct from
    stored coefficients.
    
    Args:
        artifact_dir: Directory containing saved artifacts
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Merged model state dictionary
    """
    import os
    
    # First, try to load pre-saved merged model
    merged_model_path = os.path.join(artifact_dir, "merged_state_dict.pt")
    if os.path.exists(merged_model_path):
        print(f"Loading pre-saved merged model from {merged_model_path}")
        merged_state_dict = torch.load(merged_model_path, map_location=device, weights_only=False)
        return merged_state_dict
    
    # If not available, reconstruct from artifacts
    print(f"No pre-saved merged model found, reconstructing from artifacts...")
    
    artifacts = load_all_artifacts(artifact_dir, device=device)
    
    bases = artifacts["bases"]
    compressed = artifacts["compressed"]
    config = artifacts["config"]
    diagnostics = artifacts["diagnostics"]
    
    # Get task names
    task_names = list(diagnostics.get("task_weights", {}).keys())
    if not task_names:
        first_param = next(iter(compressed.values()))
        task_names = list(first_param.keys())
    
    # Get weights
    if "task_weights" in diagnostics:
        weights = diagnostics["task_weights"]
    else:
        weights = compute_uniform_weights(task_names)
    
    # Get shapes
    original_shapes = {}
    for param_name, param_diag in diagnostics.get("per_parameter", {}).items():
        if "original_shape" in param_diag:
            original_shapes[param_name] = torch.Size(param_diag["original_shape"])
    
    masks = {}
    
    # Merge parameters
    merged_deltas = merge_all_parameters(
        compressed,
        bases,
        masks,
        weights,
        original_shapes,
        config,
        device=device
    )
    
    # Load base model
    base_model_path = config.get("base_model_path", "")
    if not base_model_path or not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model path not found in config or doesn't exist: {base_model_path}. "
            "Please provide base model path in artifacts config or use reconstruct_from_artifacts instead."
        )
    
    base_state_dict = load_checkpoint(base_model_path, device=device)
    merged_state_dict = apply_merged_deltas(base_state_dict, merged_deltas, device=device)
    
    return merged_state_dict


def reconstruct_from_artifacts(
    artifact_dir: str,
    base_model_path: str,
    output_path: str,
    device: str = "cpu"
) -> Dict:
    """
    Reconstruct merged model from stored artifacts.
    
    This function loads compressed bases, coefficients, and other artifacts
    to reconstruct the merged model without needing the original finetuned
    checkpoints. This enables deployment and sharing without distributing
    all task-specific models.
    
    Args:
        artifact_dir: Directory containing saved artifacts
        base_model_path: Path to base model checkpoint
        output_path: Path to save reconstructed merged model
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary with:
            - merged_state_dict: Reconstructed merged model
            - diagnostics: Loaded diagnostics
            - config: Loaded configuration
    """
    print(f"[1/5] Loading artifacts from {artifact_dir}...")
    artifacts = load_all_artifacts(artifact_dir, device=device)
    
    bases = artifacts["bases"]
    compressed = artifacts["compressed"]
    config = artifacts["config"]
    diagnostics = artifacts["diagnostics"]
    
    print(f"      Loaded {len(bases)} parameter bases")
    
    # Get task names from diagnostics
    task_names = list(diagnostics.get("task_weights", {}).keys())
    if not task_names:
        # Fallback: extract from compressed data
        first_param = next(iter(compressed.values()))
        task_names = list(first_param.keys())
    
    print(f"      Found {len(task_names)} tasks: {task_names}")
    
    # Compute weights (use stored weights if available, otherwise uniform)
    print(f"[2/5] Computing task weights...")
    if "task_weights" in diagnostics:
        weights = diagnostics["task_weights"]
        print(f"      Using stored task weights")
    else:
        weights = compute_uniform_weights(task_names)
        print(f"      Using uniform weights")
    
    # Get original shapes from diagnostics
    print(f"[3/5] Extracting parameter shapes...")
    original_shapes = {}
    for param_name, param_diag in diagnostics.get("per_parameter", {}).items():
        if "original_shape" in param_diag:
            original_shapes[param_name] = torch.Size(param_diag["original_shape"])
    print(f"      Found shapes for {len(original_shapes)} parameters")
    
    # Load masks if needed (not stored in artifacts currently)
    masks = {}
    
    # Merge parameters
    print(f"[4/5] Reconstructing merged parameters...")
    merged_deltas = merge_all_parameters(
        compressed,
        bases,
        masks,
        weights,
        original_shapes,
        config,
        device=device
    )
    print(f"      Merged {len(merged_deltas)} parameters")
    
    # Load base model and apply deltas
    print(f"[5/5] Loading base model and applying deltas...")
    base_state_dict = load_checkpoint(base_model_path, device=device)
    
    merged_state_dict = apply_merged_deltas(base_state_dict, merged_deltas, device=device)
    
    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    torch.save(merged_state_dict, output_path)
    
    print("\n" + "="*60)
    print("Reconstruction complete!")
    print("="*60)
    
    return {
        "merged_state_dict": merged_state_dict,
        "diagnostics": diagnostics,
        "config": config
    }


def main():
    """CLI entry point for artifact reload."""
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
