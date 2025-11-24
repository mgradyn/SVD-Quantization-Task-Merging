"""
Reload and verify SVD-Hybrid merged model from saved artifacts.

This script reconstructs a merged model from stored artifacts without
requiring access to the original finetuned checkpoints.
"""
import argparse
import os
import sys
import json
import torch
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.svd_hybrid.reload import reload_merged_model_from_artifacts
from src.svd_hybrid.storage import load_merged_model


def compute_state_dict_checksum(state_dict: dict) -> str:
    """
    Compute checksum of state dict for verification.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        MD5 checksum string
    """
    # Concatenate all parameter values and compute hash
    hasher = hashlib.md5()
    
    for key in sorted(state_dict.keys()):
        param = state_dict[key]
        if isinstance(param, torch.Tensor):
            hasher.update(param.cpu().numpy().tobytes())
    
    return hasher.hexdigest()


def verify_reconstruction(
    artifact_dir: str,
    merged_model_path: str,
    verbose: bool = True
) -> bool:
    """
    Verify that reloaded model matches saved merged model.
    
    Args:
        artifact_dir: Directory containing artifacts
        merged_model_path: Path to saved merged model
        verbose: Print detailed comparison
        
    Returns:
        True if models match
    """
    print("\n" + "="*60)
    print("Verifying Reconstruction")
    print("="*60)
    
    # Load saved merged model
    print(f"\nLoading saved merged model from {merged_model_path}")
    saved_merged = load_merged_model(merged_model_path)
    saved_checksum = compute_state_dict_checksum(saved_merged)
    
    # Reload from artifacts
    print(f"Reloading model from artifacts in {artifact_dir}")
    reloaded_merged = reload_merged_model_from_artifacts(artifact_dir)
    reloaded_checksum = compute_state_dict_checksum(reloaded_merged)
    
    # Compare checksums
    checksums_match = (saved_checksum == reloaded_checksum)
    
    if verbose:
        print(f"\nSaved model checksum:    {saved_checksum}")
        print(f"Reloaded model checksum: {reloaded_checksum}")
        print(f"Checksums match: {checksums_match}")
    
    # Compare parameter-by-parameter
    if verbose and not checksums_match:
        print("\nParameter-wise comparison:")
        all_keys = set(saved_merged.keys()) | set(reloaded_merged.keys())
        
        for key in sorted(all_keys):
            if key not in saved_merged:
                print(f"  {key}: MISSING in saved model")
            elif key not in reloaded_merged:
                print(f"  {key}: MISSING in reloaded model")
            else:
                saved_param = saved_merged[key]
                reloaded_param = reloaded_merged[key]
                
                if saved_param.shape != reloaded_param.shape:
                    print(f"  {key}: SHAPE MISMATCH ({saved_param.shape} vs {reloaded_param.shape})")
                else:
                    max_diff = (saved_param - reloaded_param).abs().max().item()
                    mean_diff = (saved_param - reloaded_param).abs().mean().item()
                    
                    if max_diff > 1e-5:
                        print(f"  {key}: DIFF max={max_diff:.6e}, mean={mean_diff:.6e}")
    
    return checksums_match


def run_evaluation(
    merged_model_path: str,
    tasks: list,
    eval_script: str = None
) -> dict:
    """
    Run evaluation on merged model.
    
    Args:
        merged_model_path: Path to merged model
        tasks: List of tasks to evaluate
        eval_script: Optional path to evaluation script
        
    Returns:
        Dictionary of evaluation results
    """
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    if eval_script is None:
        print("No evaluation script specified, skipping evaluation")
        return {}
    
    # This would call external evaluation code
    # For now, just placeholder
    print(f"Would evaluate {merged_model_path} on tasks: {tasks}")
    print("Evaluation not implemented in this script")
    
    return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reload SVD-Hybrid merged model from artifacts"
    )
    
    parser.add_argument("--artifact-dir", type=str, required=True,
                       help="Directory containing saved artifacts")
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save reloaded merged model (optional)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify against original merged model")
    parser.add_argument("--merged-model-path", type=str, default=None,
                       help="Path to original merged model for verification")
    parser.add_argument("--eval", action="store_true",
                       help="Run evaluation after reloading")
    parser.add_argument("--eval-script", type=str, default=None,
                       help="Path to evaluation script")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    # Check artifact directory exists
    if not os.path.exists(args.artifact_dir):
        print(f"Error: Artifact directory not found: {args.artifact_dir}")
        return 1
    
    # Load config from artifacts
    config_path = os.path.join(args.artifact_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        if args.verbose:
            print("\nLoaded configuration:")
            print(json.dumps(config, indent=2))
    else:
        print(f"Warning: No config.json found in {args.artifact_dir}")
        config = {}
    
    # Reload model from artifacts
    print("\n" + "="*60)
    print("Reloading Model from Artifacts")
    print("="*60)
    print(f"Artifact directory: {args.artifact_dir}")
    
    try:
        merged_state_dict = reload_merged_model_from_artifacts(args.artifact_dir)
        print(f"\nSuccessfully reloaded model with {len(merged_state_dict)} parameters")
        
        # Print some statistics
        if args.verbose:
            total_params = sum(p.numel() for p in merged_state_dict.values() if isinstance(p, torch.Tensor))
            print(f"Total parameters: {total_params:,}")
            
            print("\nParameter shapes:")
            for key in sorted(merged_state_dict.keys())[:10]:  # Show first 10
                if isinstance(merged_state_dict[key], torch.Tensor):
                    print(f"  {key}: {merged_state_dict[key].shape}")
            if len(merged_state_dict) > 10:
                print(f"  ... and {len(merged_state_dict) - 10} more")
    
    except Exception as e:
        print(f"\nError reloading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save reloaded model if requested
    if args.output_path:
        print(f"\nSaving reloaded model to {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        torch.save(merged_state_dict, args.output_path)
        print("Model saved successfully")
    
    # Verify against original if requested
    if args.verify:
        if not args.merged_model_path:
            print("\nError: --merged-model-path required for verification")
            return 1
        
        if not os.path.exists(args.merged_model_path):
            print(f"\nError: Merged model not found: {args.merged_model_path}")
            return 1
        
        match = verify_reconstruction(
            args.artifact_dir,
            args.merged_model_path,
            verbose=args.verbose
        )
        
        if match:
            print("\n✓ Verification PASSED: Reloaded model matches original")
        else:
            print("\n✗ Verification FAILED: Reloaded model differs from original")
            return 1
    
    # Run evaluation if requested
    if args.eval:
        tasks = config.get('tasks', [])
        results = run_evaluation(
            args.output_path or "reloaded_model.pt",
            tasks,
            args.eval_script
        )
        
        if results:
            print("\nEvaluation Results:")
            print(json.dumps(results, indent=2))
    
    print("\n" + "="*60)
    print("Reload Complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
