"""
Integration test for SVD-Hybrid pipeline with synthetic data.
"""
import torch
import pytest
import tempfile
import os
from src.svd_hybrid.config import SVDHybridConfig
from src.svd_hybrid.cli import run_svd_hybrid_pipeline


def create_synthetic_checkpoint(dim=100, seed=42):
    """Create a synthetic model checkpoint."""
    torch.manual_seed(seed)
    return {
        "layer1.weight": torch.randn(dim, dim),
        "layer2.weight": torch.randn(dim, dim // 2),
        "layer3.weight": torch.randn(dim // 2, dim // 4)
    }


def create_synthetic_mask(checkpoint, sparsity=0.5, seed=42):
    """Create a synthetic mask."""
    torch.manual_seed(seed)
    mask = {}
    for key, param in checkpoint.items():
        mask[key] = torch.rand_like(param) > sparsity
    return mask


@pytest.mark.slow
def test_svd_hybrid_pipeline_basic():
    """Test basic SVD-Hybrid pipeline with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic base model
        base_model = create_synthetic_checkpoint(dim=50, seed=0)
        base_path = os.path.join(tmpdir, "base_model.pt")
        torch.save(base_model, base_path)
        
        # Create synthetic fine-tuned models (4 tasks)
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir)
        
        tasks = ["task1", "task2", "task3", "task4"]
        for i, task in enumerate(tasks):
            # Create task model with small deltas
            task_model = {}
            for key, param in base_model.items():
                delta = torch.randn_like(param) * 0.1
                task_model[key] = param + delta
            
            task_path = os.path.join(checkpoint_dir, f"{task}.pt")
            torch.save(task_model, task_path)
        
        # Create config
        config = SVDHybridConfig(
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            base_model_path=base_path,
            mask_dir="",  # No masks
            svd_energy_threshold=0.90,
            svd_max_rank=32,
            svd_center=True,
            svd_fp16=False,  # Use FP32 for testing
            svd_low_bits=4,
            svd_rtvq_stages=2,
            svd_weighting="uniform",
            svd_store_artifacts=True,
            svd_eval_reconstruction=True,
            output_dir=tmpdir,
            artifact_dir=os.path.join(tmpdir, "artifacts"),
            device="cpu"
        )
        
        # Run pipeline
        results = run_svd_hybrid_pipeline(config)
        
        # Check results
        assert "merged_state_dict" in results
        assert "diagnostics" in results
        assert "bases" in results
        assert "compressed" in results
        
        # Check merged model has same keys as base
        merged_state_dict = results["merged_state_dict"]
        assert set(merged_state_dict.keys()) == set(base_model.keys())
        
        # Check diagnostics
        diagnostics = results["diagnostics"]
        assert "summary" in diagnostics
        assert "per_parameter" in diagnostics
        
        # Check summary metrics
        summary = diagnostics["summary"]
        assert "average_rank" in summary
        assert "average_energy_retained" in summary
        assert "average_reconstruction_error" in summary
        assert "average_compression_ratio" in summary
        
        # Energy retained should be close to threshold
        assert summary["average_energy_retained"] >= config.svd_energy_threshold - 0.05
        
        # Reconstruction error should be reasonable (higher for synthetic small data with quantization)
        assert summary["average_reconstruction_error"] < 1.0
        
        # Compression ratio should be computed (may be < 1 for small synthetic data)
        assert summary["average_compression_ratio"] > 0
        
        # Check artifacts were saved
        artifact_dir = config.artifact_dir
        assert os.path.exists(os.path.join(artifact_dir, "diagnostics.json"))
        assert os.path.exists(os.path.join(artifact_dir, "config.json"))
        
        # Check merged model was saved
        assert os.path.exists(os.path.join(config.output_dir, "merged_state_dict.pt"))


@pytest.mark.slow
def test_svd_hybrid_pipeline_with_masks():
    """Test SVD-Hybrid pipeline with masks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic base model
        base_model = create_synthetic_checkpoint(dim=50, seed=0)
        base_path = os.path.join(tmpdir, "base_model.pt")
        torch.save(base_model, base_path)
        
        # Create synthetic fine-tuned models
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(checkpoint_dir)
        
        tasks = ["task1", "task2"]
        for i, task in enumerate(tasks):
            task_model = {}
            for key, param in base_model.items():
                delta = torch.randn_like(param) * 0.1
                task_model[key] = param + delta
            
            task_path = os.path.join(checkpoint_dir, f"{task}.pt")
            torch.save(task_model, task_path)
        
        # Create masks
        mask_dir = os.path.join(tmpdir, "masks")
        os.makedirs(mask_dir)
        
        for task in tasks:
            mask = create_synthetic_mask(base_model, sparsity=0.7)
            mask_path = os.path.join(mask_dir, f"{task}_mask.pt")
            torch.save(mask, mask_path)
        
        # Create config with masks
        config = SVDHybridConfig(
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            base_model_path=base_path,
            mask_dir=mask_dir,
            svd_energy_threshold=0.85,
            svd_max_rank=16,
            svd_mask_strategy="union",
            svd_weighting="uniform",
            svd_store_artifacts=False,
            svd_eval_reconstruction=True,
            output_dir=tmpdir,
            artifact_dir=os.path.join(tmpdir, "artifacts"),
            device="cpu"
        )
        
        # Run pipeline
        results = run_svd_hybrid_pipeline(config)
        
        # Check results
        assert "merged_state_dict" in results
        merged_state_dict = results["merged_state_dict"]
        assert set(merged_state_dict.keys()) == set(base_model.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
