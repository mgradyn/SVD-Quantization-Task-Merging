"""
Tests for quantization_utils module.
"""
import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import quantization_utils


def test_absmax_quantization_8bit():
    """Test absmax quantization with 8 bits."""
    X = torch.randn(100)
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=8)
    
    # Check output types
    assert X_q.dtype == torch.int8
    assert isinstance(scale, torch.Tensor)
    
    # Check quantized values are in valid range
    assert X_q.min() >= -128
    assert X_q.max() <= 127


def test_absmax_quantization_16bit():
    """Test absmax quantization with 16 bits."""
    X = torch.randn(50)
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=16)
    
    assert X_q.dtype == torch.int16
    assert X_q.min() >= -32768
    assert X_q.max() <= 32767


def test_asymmetric_quantization_basic():
    """Test asymmetric quantization."""
    X = torch.randn(100)
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=8)
    
    # Check output types
    assert X_q.dtype == torch.uint8
    assert isinstance(scale, torch.Tensor)
    assert isinstance(zero_point, torch.Tensor)
    
    # Check quantized values are in valid range
    assert X_q.min() >= 0
    assert X_q.max() <= 255


def test_asymmetric_quantization_zero_point_formula():
    """Test that zero_point follows reference formula: -round(scale * X_min)."""
    X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=4)
    
    # Verify formula
    X_min = X.min()
    X_max = X.max()
    qmin, qmax = 0, 15
    
    expected_scale = (qmax - qmin) / (X_max - X_min)
    expected_zero_point = -torch.round(expected_scale * X_min)
    expected_zero_point = torch.clamp(expected_zero_point, qmin, qmax)
    
    assert torch.allclose(scale, expected_scale, rtol=1e-5)
    assert torch.allclose(zero_point, expected_zero_point, rtol=1e-5)


def test_dequantize_absmax():
    """Test absmax dequantization reconstructs values."""
    X = torch.randn(50)
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=8)
    X_recon = quantization_utils.dequantize_absmax(X_q, scale)
    
    # Check shape matches
    assert X_recon.shape == X.shape
    
    # Check reconstruction quality (should be close but not exact due to quantization)
    error = (X - X_recon).abs().max()
    assert error < scale.item()


def test_dequantize_asymmetric():
    """Test asymmetric dequantization reconstructs values."""
    X = torch.randn(50)
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=8)
    X_recon = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
    
    # Check shape matches
    assert X_recon.shape == X.shape
    
    # Check reconstruction quality
    relative_error = (X - X_recon).norm() / X.norm()
    assert relative_error < 0.1  # Should be reasonably accurate


def test_quantization_error_check():
    """Test error checking for absmax quantization."""
    X = torch.randn(100)
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=8)
    
    metrics = quantization_utils.quantization_error_check(X, X_q, scale, verbose=False)
    
    # Check all metrics are present
    assert "l1_error" in metrics
    assert "l1_relative" in metrics
    assert "l2_error" in metrics
    assert "l2_relative" in metrics
    assert "max_error" in metrics
    
    # Check metrics are reasonable
    assert metrics["l1_relative"] >= 0
    assert metrics["l2_relative"] >= 0


def test_quantization_error_check_asymmetric():
    """Test error checking for asymmetric quantization."""
    X = torch.randn(100)
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=8)
    
    metrics = quantization_utils.quantization_error_check_asymmetric(
        X, X_q, scale, zero_point, verbose=False
    )
    
    # Check all metrics are present
    assert "l1_error" in metrics
    assert "l1_relative" in metrics
    assert "l2_error" in metrics
    assert "l2_relative" in metrics
    assert "max_error" in metrics


def test_constant_tensor():
    """Test quantization of constant tensor."""
    X = torch.ones(50) * 3.14
    
    # Absmax - all values should be the same
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=8)
    assert X_q.float().std() < 1e-6  # All quantized to same value (127 for positive constant)
    
    # Asymmetric - when handled as constant, should return all zeros
    X_zero = torch.ones(50) * 0.0
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X_zero, qbit=8)
    assert X_q.float().std() < 1e-6  # Should all be zero


def test_empty_tensor():
    """Test handling of empty tensors."""
    X = torch.tensor([])
    
    # Absmax
    X_q, scale = quantization_utils.absmax_quantization(X, qbit=8)
    assert X_q.numel() == 0
    
    # Asymmetric
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=8)
    assert X_q.numel() == 0


def test_different_bit_widths():
    """Test that more bits give better reconstruction."""
    X = torch.randn(100)
    
    # Test absmax
    X_q2, s2 = quantization_utils.absmax_quantization(X, qbit=8)
    X_q4, s4 = quantization_utils.absmax_quantization(X, qbit=16)
    
    r2 = quantization_utils.dequantize_absmax(X_q2, s2)
    r4 = quantization_utils.dequantize_absmax(X_q4, s4)
    
    error2 = (X - r2).norm()
    error4 = (X - r4).norm()
    
    # 16 bits should be more accurate than 8 bits
    assert error4 < error2


def test_compression_ratio():
    """Test compression ratio calculation."""
    X = torch.randn(100)
    
    # 8-bit quantization
    ratio8 = quantization_utils.compute_compression_ratio(X, qbit=8, method="asymmetric")
    
    # 4-bit quantization
    ratio4 = quantization_utils.compute_compression_ratio(X, qbit=4, method="asymmetric")
    
    # Lower bits should give higher compression
    assert ratio4 > ratio8
    
    # Should be reasonable ratios
    assert ratio8 > 1.0
    assert ratio4 > ratio8


def test_negative_values():
    """Test quantization handles negative values correctly."""
    X = torch.randn(100) - 2.0  # Shift to be mostly negative
    
    # Asymmetric should handle this
    X_q, scale, zero_point = quantization_utils.asymmetric_quantization(X, qbit=8)
    X_recon = quantization_utils.dequantize_asymmetric(X_q, scale, zero_point)
    
    # Check that negative values are preserved
    assert X_recon.min() < 0
    
    # Check reconstruction quality
    relative_error = (X - X_recon).norm() / X.norm()
    assert relative_error < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
