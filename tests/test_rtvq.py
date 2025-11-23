"""
Tests for RTVQ (Residual Task Vector Quantization).
"""
import torch
import pytest
from src.svd_hybrid.rtvq import (
    RTVQQuantizer,
    asymmetric_quantization,
    asymmetric_dequantization,
    multistage_residual_quantization,
    multistage_residual_dequantization,
    compute_quantization_error
)


def test_asymmetric_quantization_basic():
    """Test basic asymmetric quantization."""
    tensor = torch.randn(100)
    num_bits = 4
    
    q_indices, scale, zero_point = asymmetric_quantization(tensor, num_bits)
    
    # Check output types
    assert q_indices.dtype == torch.uint8
    assert scale.ndim == 0  # scalar
    assert zero_point.ndim == 0  # scalar
    
    # Check quantized values are in valid range
    n_levels = 2 ** num_bits
    assert q_indices.min() >= 0
    assert q_indices.max() < n_levels


def test_asymmetric_dequantization():
    """Test asymmetric dequantization reconstructs values."""
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    num_bits = 4
    
    q_indices, scale, zero_point = asymmetric_quantization(tensor, num_bits)
    dequantized = asymmetric_dequantization(q_indices, scale, zero_point)
    
    # Check reconstruction is close to original
    error = (tensor - dequantized).abs().max()
    
    # With 4 bits, we have 16 levels, so max error should be roughly scale/2
    expected_max_error = scale.item() / 2 * 1.5  # Add some tolerance
    assert error <= expected_max_error


def test_multistage_quantization():
    """Test multi-stage residual quantization."""
    tensor = torch.randn(50)
    num_bits = 2
    num_stages = 3
    
    payloads = multistage_residual_quantization(tensor, num_bits, num_stages)
    
    assert len(payloads) == num_stages
    
    # Check each stage has required fields
    for i, payload in enumerate(payloads):
        assert payload["stage"] == i
        assert "quantized" in payload
        assert "scale" in payload
        assert "zero_point" in payload
        assert "residual_norm" in payload


def test_multistage_reconstruction():
    """Test multi-stage dequantization improves reconstruction."""
    tensor = torch.randn(100)
    num_bits = 2
    
    # Single stage
    payloads_1 = multistage_residual_quantization(tensor, num_bits, num_stages=1)
    recon_1 = multistage_residual_dequantization(payloads_1)
    error_1 = (tensor - recon_1).norm() / tensor.norm()
    
    # Two stages
    payloads_2 = multistage_residual_quantization(tensor, num_bits, num_stages=2)
    recon_2 = multistage_residual_dequantization(payloads_2)
    error_2 = (tensor - recon_2).norm() / tensor.norm()
    
    # More stages should give better reconstruction
    assert error_2 < error_1


def test_rtvq_quantizer_class():
    """Test RTVQQuantizer class interface."""
    quantizer = RTVQQuantizer(num_bits=4, num_stages=2)
    
    tensor = torch.randn(100)
    
    # Quantize
    quantized_obj = quantizer.quantize(tensor)
    
    # Check structure
    assert "payloads" in quantized_obj
    assert "num_bits" in quantized_obj
    assert "num_stages" in quantized_obj
    assert len(quantized_obj["payloads"]) == 2
    
    # Dequantize
    reconstructed = quantizer.dequantize(quantized_obj)
    
    # Check shape matches
    assert reconstructed.shape == tensor.shape
    
    # Check reconstruction quality
    relative_error = (tensor - reconstructed).norm() / tensor.norm()
    assert relative_error < 0.5  # Should reconstruct reasonably well


def test_quantization_error_metrics():
    """Test quantization error computation."""
    original = torch.randn(50)
    
    quantizer = RTVQQuantizer(num_bits=4, num_stages=2)
    quantized_obj = quantizer.quantize(original)
    
    error_metrics = compute_quantization_error(original, quantized_obj)
    
    # Check all metrics are present
    assert "absolute_error" in error_metrics
    assert "relative_error" in error_metrics
    assert "max_absolute_error" in error_metrics
    assert "mean_absolute_error" in error_metrics
    
    # Check values are reasonable
    assert error_metrics["relative_error"] >= 0
    assert error_metrics["relative_error"] < 1.0  # Should be reasonable


def test_different_bit_widths():
    """Test that more bits give better reconstruction."""
    tensor = torch.randn(100)
    
    quantizer_2bit = RTVQQuantizer(num_bits=2, num_stages=1)
    quantizer_4bit = RTVQQuantizer(num_bits=4, num_stages=1)
    quantizer_8bit = RTVQQuantizer(num_bits=8, num_stages=1)
    
    q2 = quantizer_2bit.quantize(tensor)
    q4 = quantizer_4bit.quantize(tensor)
    q8 = quantizer_8bit.quantize(tensor)
    
    r2 = quantizer_2bit.dequantize(q2)
    r4 = quantizer_4bit.dequantize(q4)
    r8 = quantizer_8bit.dequantize(q8)
    
    error2 = (tensor - r2).norm()
    error4 = (tensor - r4).norm()
    error8 = (tensor - r8).norm()
    
    # More bits should give better reconstruction
    assert error8 < error4 < error2


def test_empty_tensor():
    """Test handling of empty tensors."""
    tensor = torch.tensor([])
    
    quantizer = RTVQQuantizer(num_bits=4, num_stages=2)
    quantized_obj = quantizer.quantize(tensor)
    
    # Should handle gracefully
    assert quantized_obj["payloads"] == []


def test_constant_tensor():
    """Test quantization of constant tensor."""
    tensor = torch.ones(50) * 3.14
    
    quantizer = RTVQQuantizer(num_bits=4, num_stages=1)
    quantized_obj = quantizer.quantize(tensor)
    reconstructed = quantizer.dequantize(quantized_obj)
    
    # Should reconstruct constant value well
    assert reconstructed.std() < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
