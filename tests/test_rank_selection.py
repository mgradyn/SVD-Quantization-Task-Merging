"""
Tests for rank selection in SVD basis construction.
"""
import torch
import pytest
from src.svd_hybrid.basis import select_rank, compute_energy_spectrum, compute_svd


def test_rank_selection_basic():
    """Test basic rank selection with synthetic singular values."""
    # Create synthetic singular value spectrum (exponentially decaying)
    S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])
    
    # Test with 90% energy threshold
    k = select_rank(S, energy_threshold=0.90, max_rank=None)
    
    # Verify k is reasonable
    assert k >= 1
    assert k <= len(S)
    
    # Verify energy retained
    energy_spectrum = compute_energy_spectrum(S)
    energy_retained = energy_spectrum[k-1].item()
    assert energy_retained >= 0.90, f"Energy retained {energy_retained} < 0.90"


def test_rank_selection_max_cap():
    """Test that max_rank cap is respected."""
    S = torch.ones(100)  # Uniform spectrum
    
    k = select_rank(S, energy_threshold=0.99, max_rank=10)
    
    assert k <= 10, f"Selected rank {k} exceeds max_rank 10"


def test_rank_selection_min_rank():
    """Test that min_rank is respected."""
    S = torch.tensor([100.0, 0.01, 0.001])  # First component dominates
    
    k = select_rank(S, energy_threshold=0.999, max_rank=None, min_rank=2)
    
    assert k >= 2, f"Selected rank {k} below min_rank 2"


def test_energy_spectrum():
    """Test energy spectrum computation."""
    S = torch.tensor([4.0, 3.0, 2.0, 1.0])
    # Energy: [16, 9, 4, 1], total = 30
    # Cumulative fractions: [16/30, 25/30, 29/30, 30/30]
    
    cum_energy = compute_energy_spectrum(S)
    
    assert len(cum_energy) == len(S)
    assert cum_energy[-1].item() == pytest.approx(1.0, abs=1e-6)
    assert all(cum_energy[i] <= cum_energy[i+1] for i in range(len(cum_energy)-1))


def test_rank_selection_zero_energy():
    """Test handling of zero/near-zero singular values."""
    S = torch.tensor([10.0, 1e-10, 1e-12])
    
    k = select_rank(S, energy_threshold=0.99, max_rank=None)
    
    # Should select only the first component
    assert k == 1


def test_rank_selection_different_thresholds():
    """Test different energy thresholds produce different ranks."""
    S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    
    k_50 = select_rank(S, energy_threshold=0.50, max_rank=None)
    k_90 = select_rank(S, energy_threshold=0.90, max_rank=None)
    k_99 = select_rank(S, energy_threshold=0.99, max_rank=None)
    
    # Higher threshold should need more components
    assert k_50 <= k_90 <= k_99


def test_compute_svd_preserves_device():
    """Test that compute_svd returns tensors on the same device as input."""
    # Create a simple matrix on CPU
    matrix = torch.randn(10, 4)
    
    # Compute SVD
    U, S, Vh = compute_svd(matrix)
    
    # All results should be on CPU (same device as input)
    assert U.device == matrix.device, f"U should be on {matrix.device}, got {U.device}"
    assert S.device == matrix.device, f"S should be on {matrix.device}, got {S.device}"
    assert Vh.device == matrix.device, f"Vh should be on {matrix.device}, got {Vh.device}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_svd_preserves_cuda_device():
    """Test that compute_svd returns tensors on CUDA when input is on CUDA."""
    # Create a simple matrix on CUDA
    matrix = torch.randn(10, 4).cuda()
    
    # Compute SVD
    U, S, Vh = compute_svd(matrix)
    
    # All results should be on CUDA (same device as input)
    assert U.device == matrix.device, f"U should be on {matrix.device}, got {U.device}"
    assert S.device == matrix.device, f"S should be on {matrix.device}, got {S.device}"
    assert Vh.device == matrix.device, f"Vh should be on {matrix.device}, got {Vh.device}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
