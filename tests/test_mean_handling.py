"""
Tests for mean vector handling in SVD-Hybrid compression/reconstruction.

This test suite validates the fix for the mean vector issue in the SVD-Hybrid
pipeline. The issue was that when `svd_center=True`, the mean vector calculated
during SVD basis construction was not being correctly handled during compression
and reconstruction phases.

The fix ensures:
1. Compression: The mean is subtracted before projecting to the SVD basis
2. Reconstruction: The mean is added back after reconstructing from coefficients
"""
import torch
import pytest
from src.svd_hybrid.basis import construct_basis
from src.svd_hybrid.rtvq import RTVQQuantizer
from src.svd_hybrid.compress import compress_single_task, project_to_basis, compress_masked_regions
from src.svd_hybrid.merge import reconstruct_from_coefficients, dequantize_and_average


class TestMeanVectorHandling:
    """Tests for mean vector handling in compression and reconstruction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample task deltas for testing."""
        torch.manual_seed(42)
        n_tasks = 4
        dim = 100
        deltas = [torch.randn(dim) for _ in range(n_tasks)]
        return deltas, n_tasks, dim
    
    @pytest.fixture
    def quantizer(self):
        """Create a quantizer for testing."""
        return RTVQQuantizer(num_bits=4, num_stages=2)
    
    def test_compress_with_mean_improves_reconstruction(self, sample_data, quantizer):
        """Test that using mean in compression significantly reduces error."""
        deltas, n_tasks, dim = sample_data
        
        # Build basis with centering
        basis = construct_basis(deltas, energy_threshold=0.95, center=True, verbose=False)
        
        assert basis["mean"] is not None, "Mean should be present when centering"
        
        task_delta = deltas[0]
        U_high = basis['U_high']
        U_low = basis['U_low']
        mean = basis['mean']
        
        # Compress WITH mean handling (the fix)
        compressed_with_mean = compress_single_task(
            task_delta, U_high, U_low, quantizer, 'cpu', mean=mean
        )
        c_high_with = compressed_with_mean['c_high_fp16'].float()
        c_low_with = quantizer.dequantize(compressed_with_mean['c_low_quant'], device='cpu').float()
        
        reconstructed_with_mean = reconstruct_from_coefficients(
            c_high_with, c_low_with, U_high, U_low, 'cpu', mean=mean
        )
        error_with_mean = (reconstructed_with_mean - task_delta).norm() / task_delta.norm()
        
        # Compress WITHOUT mean handling (the old buggy behavior)
        compressed_without_mean = compress_single_task(
            task_delta, U_high, U_low, quantizer, 'cpu', mean=None
        )
        c_high_without = compressed_without_mean['c_high_fp16'].float()
        c_low_without = quantizer.dequantize(compressed_without_mean['c_low_quant'], device='cpu').float()
        
        reconstructed_without_mean = reconstruct_from_coefficients(
            c_high_without, c_low_without, U_high, U_low, 'cpu', mean=None
        )
        error_without_mean = (reconstructed_without_mean - task_delta).norm() / task_delta.norm()
        
        # The error with mean handling should be much lower
        assert error_with_mean < error_without_mean, \
            f"Error with mean ({error_with_mean:.6f}) should be less than without ({error_without_mean:.6f})"
        
        # The error with mean handling should be very low (close to quantization error)
        assert error_with_mean < 0.01, \
            f"Error with mean handling ({error_with_mean:.6f}) should be < 1%"
        
        # The improvement should be significant (at least 10x better)
        improvement = error_without_mean / error_with_mean
        assert improvement > 10, \
            f"Mean handling should improve error by at least 10x, got {improvement:.1f}x"
    
    def test_no_centering_mean_is_none(self, sample_data, quantizer):
        """Test that without centering, mean is None and reconstruction works."""
        deltas, n_tasks, dim = sample_data
        
        # Build basis without centering
        basis = construct_basis(deltas, energy_threshold=0.95, center=False, verbose=False)
        
        assert basis["mean"] is None, "Mean should be None when not centering"
        
        task_delta = deltas[0]
        U_high = basis['U_high']
        U_low = basis['U_low']
        
        # Compress and reconstruct
        compressed = compress_single_task(
            task_delta, U_high, U_low, quantizer, 'cpu', mean=None
        )
        c_high = compressed['c_high_fp16'].float()
        c_low = quantizer.dequantize(compressed['c_low_quant'], device='cpu').float()
        
        reconstructed = reconstruct_from_coefficients(
            c_high, c_low, U_high, U_low, 'cpu', mean=None
        )
        
        error = (reconstructed - task_delta).norm() / task_delta.norm()
        
        # Error should be low (dominated by quantization)
        assert error < 0.05, f"Error ({error:.6f}) should be < 5%"
    
    def test_mean_shape_handling(self, sample_data, quantizer):
        """Test that mean vector shape handling works for both [D] and [D x 1]."""
        deltas, n_tasks, dim = sample_data
        
        basis = construct_basis(deltas, energy_threshold=0.95, center=True, verbose=False)
        
        task_delta = deltas[0]
        U_high = basis['U_high']
        U_low = basis['U_low']
        mean = basis['mean']  # This should be [D x 1]
        
        # Test with [D x 1] shape (as returned by construct_basis)
        compressed = compress_single_task(
            task_delta, U_high, U_low, quantizer, 'cpu', mean=mean
        )
        c_high = compressed['c_high_fp16'].float()
        c_low = quantizer.dequantize(compressed['c_low_quant'], device='cpu').float()
        
        reconstructed = reconstruct_from_coefficients(
            c_high, c_low, U_high, U_low, 'cpu', mean=mean
        )
        error_2d = (reconstructed - task_delta).norm() / task_delta.norm()
        
        # Test with [D] shape (squeezed)
        mean_squeezed = mean.squeeze()
        compressed_sq = compress_single_task(
            task_delta, U_high, U_low, quantizer, 'cpu', mean=mean_squeezed
        )
        c_high_sq = compressed_sq['c_high_fp16'].float()
        c_low_sq = quantizer.dequantize(compressed_sq['c_low_quant'], device='cpu').float()
        
        reconstructed_sq = reconstruct_from_coefficients(
            c_high_sq, c_low_sq, U_high, U_low, 'cpu', mean=mean_squeezed
        )
        error_1d = (reconstructed_sq - task_delta).norm() / task_delta.norm()
        
        # Both should give similar results
        assert abs(error_2d - error_1d) < 1e-6, \
            f"Shape handling should be consistent: 2D error={error_2d:.6f}, 1D error={error_1d:.6f}"
    
    def test_compress_masked_regions_uses_mean(self, sample_data, quantizer):
        """Test that compress_masked_regions correctly extracts and uses mean."""
        deltas, n_tasks, dim = sample_data
        
        # Build basis with centering
        basis = construct_basis(deltas, energy_threshold=0.95, center=True, verbose=False)
        
        # Wrap in the expected format for masked regions
        basis_masked = basis
        
        # Create task deltas dict
        task_deltas_masked = {f"task{i}": d for i, d in enumerate(deltas)}
        
        # Compress using compress_masked_regions
        compressed = compress_masked_regions(
            task_deltas_masked,
            None,  # no unmasked deltas
            basis_masked,
            None,  # no unmasked basis
            quantizer,
            'cpu'
        )
        
        # Verify each task was compressed
        assert len(compressed) == n_tasks
        
        # For each task, verify reconstruction is accurate
        for i, task_name in enumerate(compressed.keys()):
            artifact = compressed[task_name]["masked"]
            c_high = artifact['c_high_fp16'].float()
            c_low = quantizer.dequantize(artifact['c_low_quant'], device='cpu').float()
            
            reconstructed = reconstruct_from_coefficients(
                c_high, c_low, 
                basis["U_high"], basis["U_low"], 
                'cpu', mean=basis["mean"]
            )
            
            original = task_deltas_masked[task_name]
            error = (reconstructed - original).norm() / original.norm()
            
            assert error < 0.01, \
                f"Reconstruction error for {task_name} ({error:.6f}) should be < 1%"


class TestReconstructFromCoefficients:
    """Tests specifically for reconstruct_from_coefficients function."""
    
    def test_reconstruct_without_mean(self):
        """Test basic reconstruction without mean."""
        torch.manual_seed(42)
        
        dim = 50
        k = 5
        
        # Create orthogonal basis
        U_full = torch.linalg.qr(torch.randn(dim, dim))[0]
        U_high = U_full[:, :k]
        U_low = U_full[:, k:]
        
        # Create coefficients
        c_high = torch.randn(k)
        c_low = torch.randn(dim - k)
        
        # Reconstruct
        delta = reconstruct_from_coefficients(c_high, c_low, U_high, U_low, 'cpu', mean=None)
        
        # Verify shape
        assert delta.shape == (dim,)
        
        # Verify reconstruction formula
        expected = U_high @ c_high + U_low @ c_low
        assert torch.allclose(delta, expected, atol=1e-5)
    
    def test_reconstruct_with_mean(self):
        """Test reconstruction with mean vector."""
        torch.manual_seed(42)
        
        dim = 50
        k = 5
        
        # Create orthogonal basis
        U_full = torch.linalg.qr(torch.randn(dim, dim))[0]
        U_high = U_full[:, :k]
        U_low = U_full[:, k:]
        
        # Create coefficients and mean
        c_high = torch.randn(k)
        c_low = torch.randn(dim - k)
        mean = torch.randn(dim)
        
        # Reconstruct
        delta = reconstruct_from_coefficients(c_high, c_low, U_high, U_low, 'cpu', mean=mean)
        
        # Verify shape
        assert delta.shape == (dim,)
        
        # Verify reconstruction formula
        expected = U_high @ c_high + U_low @ c_low + mean
        assert torch.allclose(delta, expected, atol=1e-5)
    
    def test_reconstruct_mean_shape_2d(self):
        """Test reconstruction with [D x 1] mean shape."""
        torch.manual_seed(42)
        
        dim = 50
        k = 5
        
        U_full = torch.linalg.qr(torch.randn(dim, dim))[0]
        U_high = U_full[:, :k]
        U_low = U_full[:, k:]
        
        c_high = torch.randn(k)
        c_low = torch.randn(dim - k)
        mean_2d = torch.randn(dim, 1)  # [D x 1] shape
        
        # Reconstruct
        delta = reconstruct_from_coefficients(c_high, c_low, U_high, U_low, 'cpu', mean=mean_2d)
        
        # Verify shape
        assert delta.shape == (dim,)
        
        # Verify reconstruction formula
        expected = U_high @ c_high + U_low @ c_low + mean_2d.squeeze()
        assert torch.allclose(delta, expected, atol=1e-5)


class TestProjectToBasis:
    """Tests for project_to_basis function."""
    
    def test_project_basic(self):
        """Test basic projection."""
        torch.manual_seed(42)
        
        dim = 50
        k = 5
        
        # Create orthogonal basis
        U_full = torch.linalg.qr(torch.randn(dim, dim))[0]
        U_high = U_full[:, :k]
        U_low = U_full[:, k:]
        
        # Create delta
        delta = torch.randn(dim)
        
        # Project
        c_high, c_low = project_to_basis(delta, U_high, U_low)
        
        # Verify shapes
        assert c_high.shape == (k,)
        assert c_low.shape == (dim - k,)
        
        # Verify roundtrip reconstruction
        reconstructed = U_high @ c_high + U_low @ c_low
        assert torch.allclose(reconstructed, delta, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
