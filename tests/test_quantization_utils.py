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



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
