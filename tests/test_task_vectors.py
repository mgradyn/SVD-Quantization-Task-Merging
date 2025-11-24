"""
Tests for task_vectors module.
"""
import torch
import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import task_vectors


def create_dummy_model_state():
    """Create a dummy model state dict for testing."""
    return {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5),
    }


def test_task_vector_creation():
    """Test creating a TaskVector from two checkpoints."""
    # Create dummy models
    pretrained = create_dummy_model_state()
    finetuned = create_dummy_model_state()
    
    # Create task vector
    tv = task_vectors.TaskVector(pretrained, finetuned, task_name="test_task")
    
    # Check that task vector has correct keys
    assert "layer1.weight" in tv.vector
    assert "layer1.bias" in tv.vector
    assert "layer2.weight" in tv.vector
    assert "layer2.bias" in tv.vector
    
    # Check shapes match
    assert tv.vector["layer1.weight"].shape == pretrained["layer1.weight"].shape


def test_task_vector_delta():
    """Test that task vector computes correct delta."""
    # Create models where finetuned = pretrained + 1.0
    pretrained = {
        "weight": torch.ones(5, 5),
    }
    finetuned = {
        "weight": torch.ones(5, 5) + 1.0,
    }
    
    tv = task_vectors.TaskVector(pretrained, finetuned)
    
    # Delta should be all 1.0
    expected_delta = torch.ones(5, 5)
    assert torch.allclose(tv.vector["weight"], expected_delta)


def test_task_vector_add():
    """Test adding two task vectors."""
    pretrained = {"weight": torch.ones(3, 3)}
    finetuned1 = {"weight": torch.ones(3, 3) + 1.0}
    finetuned2 = {"weight": torch.ones(3, 3) + 2.0}
    
    tv1 = task_vectors.TaskVector(pretrained, finetuned1, task_name="task1")
    tv2 = task_vectors.TaskVector(pretrained, finetuned2, task_name="task2")
    
    tv_sum = tv1 + tv2
    
    # Sum of deltas: 1.0 + 2.0 = 3.0
    expected = torch.ones(3, 3) * 3.0
    assert torch.allclose(tv_sum.vector["weight"], expected)


def test_task_vector_subtract():
    """Test subtracting two task vectors."""
    pretrained = {"weight": torch.ones(3, 3)}
    finetuned1 = {"weight": torch.ones(3, 3) + 2.0}
    finetuned2 = {"weight": torch.ones(3, 3) + 1.0}
    
    tv1 = task_vectors.TaskVector(pretrained, finetuned1)
    tv2 = task_vectors.TaskVector(pretrained, finetuned2)
    
    tv_diff = tv1 - tv2
    
    # Difference: 2.0 - 1.0 = 1.0
    expected = torch.ones(3, 3)
    assert torch.allclose(tv_diff.vector["weight"], expected)


def test_task_vector_multiply():
    """Test multiplying task vector by scalar."""
    pretrained = {"weight": torch.ones(3, 3)}
    finetuned = {"weight": torch.ones(3, 3) + 2.0}
    
    tv = task_vectors.TaskVector(pretrained, finetuned)
    tv_scaled = tv * 0.5
    
    # Delta was 2.0, scaled by 0.5 = 1.0
    expected = torch.ones(3, 3)
    assert torch.allclose(tv_scaled.vector["weight"], expected)


def test_task_vector_apply_to():
    """Test applying task vector to pretrained model."""
    pretrained = {"weight": torch.ones(3, 3)}
    finetuned = {"weight": torch.ones(3, 3) + 2.0}
    
    tv = task_vectors.TaskVector(pretrained, finetuned)
    
    # Apply task vector to pretrained
    result = tv.apply_to(pretrained)
    
    # Should recover finetuned model
    assert torch.allclose(result["weight"], finetuned["weight"])


def test_quantized_task_vector_asymmetric():
    """Test QuantizedTaskVector with asymmetric quantization."""
    import quantization_utils
    
    # Create task vector
    pretrained = {"weight": torch.randn(5, 5)}
    finetuned = {"weight": torch.randn(5, 5)}
    tv = task_vectors.TaskVector(pretrained, finetuned)
    
    # Quantize deltas
    quantized_deltas = {}
    for key, delta in tv.vector.items():
        X_q, scale, zero_point = quantization_utils.asymmetric_quantization(delta, qbit=8)
        quantized_deltas[key] = {
            "quantized": X_q,
            "scale": scale,
            "zero_point": zero_point
        }
    
    # Create quantized task vector
    qtv = task_vectors.QuantizedTaskVector(quantized_deltas, method="asymmetric")
    
    # Dequantize
    dequantized = qtv.dequantize()
    
    # Check keys match
    assert set(dequantized.keys()) == set(tv.vector.keys())
    
    # Check reconstruction is close
    for key in tv.vector.keys():
        relative_error = (tv.vector[key] - dequantized[key]).norm() / tv.vector[key].norm()
        assert relative_error < 0.1


def test_quantized_task_vector_apply():
    """Test applying quantized task vector to model."""
    import quantization_utils
    
    # Use non-constant tensors to avoid degenerate quantization
    pretrained = {"weight": torch.randn(3, 3)}
    finetuned = {"weight": pretrained["weight"] + torch.randn(3, 3) * 0.5}
    tv = task_vectors.TaskVector(pretrained, finetuned)
    
    # Quantize
    quantized_deltas = {}
    for key, delta in tv.vector.items():
        X_q, scale, zero_point = quantization_utils.asymmetric_quantization(delta, qbit=8)
        quantized_deltas[key] = {
            "quantized": X_q,
            "scale": scale,
            "zero_point": zero_point
        }
    
    qtv = task_vectors.QuantizedTaskVector(quantized_deltas, method="asymmetric")
    
    # Apply to pretrained
    result = qtv.apply_to(pretrained)
    
    # Should be close to finetuned
    relative_error = (result["weight"] - finetuned["weight"]).norm() / finetuned["weight"].norm()
    assert relative_error < 0.1


def test_quantized_finetuned_model():
    """Test QuantizedFinetunedModel."""
    finetuned = {
        "weight": torch.randn(5, 5),
        "bias": torch.randn(5)
    }
    
    # Create quantized model
    qfm = task_vectors.QuantizedFinetunedModel(finetuned, qbit=8, method="asymmetric")
    
    # Dequantize
    reconstructed = qfm.dequantize()
    
    # Check keys match
    assert set(reconstructed.keys()) == set(finetuned.keys())
    
    # Check shapes match
    assert reconstructed["weight"].shape == finetuned["weight"].shape
    assert reconstructed["bias"].shape == finetuned["bias"].shape
    
    # Check reconstruction quality
    for key in finetuned.keys():
        relative_error = (finetuned[key] - reconstructed[key]).norm() / finetuned[key].norm()
        assert relative_error < 0.1


def test_quantized_finetuned_model_get_task_vector():
    """Test getting task vector from quantized finetuned model."""
    # Use non-constant tensors
    pretrained = {"weight": torch.randn(5, 5)}
    finetuned = {"weight": pretrained["weight"] + torch.randn(5, 5) * 0.5}
    
    # Create quantized finetuned model
    qfm = task_vectors.QuantizedFinetunedModel(finetuned, qbit=8, method="asymmetric")
    
    # Get task vector
    tv = qfm.get_task_vector(pretrained)
    
    # Should be close to original delta
    expected_delta = finetuned["weight"] - pretrained["weight"]
    relative_error = (tv["weight"] - expected_delta).norm() / expected_delta.norm()
    assert relative_error < 0.2


def test_quantized_base_and_task_vector():
    """Test QuantizedBaseAndTaskVector."""
    pretrained = {"weight": torch.randn(5, 5)}
    finetuned = {"weight": torch.randn(5, 5)}
    
    tv = task_vectors.TaskVector(pretrained, finetuned)
    
    # Create quantized base and task vector
    qbatv = task_vectors.QuantizedBaseAndTaskVector(
        pretrained, tv, base_qbit=8, task_qbit=8, method="asymmetric"
    )
    
    # Dequantize (should reconstruct finetuned model)
    reconstructed = qbatv.dequantize()
    
    # Check keys match
    assert set(reconstructed.keys()) == set(finetuned.keys())
    
    # Check reconstruction quality
    for key in finetuned.keys():
        relative_error = (finetuned[key] - reconstructed[key]).norm() / finetuned[key].norm()
        assert relative_error < 0.2


def test_task_vector_skip_int64():
    """Test that int64 parameters are skipped."""
    pretrained = {
        "weight": torch.randn(3, 3),
        "buffer": torch.ones(10, dtype=torch.int64)
    }
    finetuned = {
        "weight": torch.randn(3, 3),
        "buffer": torch.ones(10, dtype=torch.int64) + 1
    }
    
    tv = task_vectors.TaskVector(pretrained, finetuned, skip_int64=True)
    
    # int64 buffer should be skipped
    assert "weight" in tv.vector
    assert "buffer" not in tv.vector


def test_task_vector_from_files():
    """Test creating TaskVector from checkpoint files."""
    # Create temporary checkpoint files
    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_path = os.path.join(tmpdir, "pretrained.pt")
        finetuned_path = os.path.join(tmpdir, "finetuned.pt")
        
        pretrained_state = {"weight": torch.ones(3, 3)}
        finetuned_state = {"weight": torch.ones(3, 3) + 1.0}
        
        torch.save(pretrained_state, pretrained_path)
        torch.save(finetuned_state, finetuned_path)
        
        # Create task vector from files
        tv = task_vectors.TaskVector(pretrained_path, finetuned_path)
        
        # Check delta
        expected_delta = torch.ones(3, 3)
        assert torch.allclose(tv.vector["weight"], expected_delta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
