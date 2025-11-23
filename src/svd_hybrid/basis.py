"""
SVD basis construction utilities with energy-based rank selection.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def stack_and_center(
    vectors: List[torch.Tensor],
    center: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Stack vectors into matrix and optionally center.
    
    Args:
        vectors: List of 1D tensors (same length)
        center: Whether to mean-center columns
        
    Returns:
        Tuple of (stacked matrix [D x N], mean vector or None)
    """
    if not vectors:
        raise ValueError("Empty vector list")
    
    # Stack into matrix [D x N]
    T = torch.stack(vectors, dim=1)
    
    mean = None
    if center:
        mean = T.mean(dim=1, keepdim=True)
        T = T - mean
    
    return T, mean


def compute_energy_spectrum(singular_values: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative energy spectrum from singular values.
    
    Args:
        singular_values: 1D tensor of singular values (sorted descending)
        
    Returns:
        Cumulative energy fractions
    """
    energy = singular_values ** 2
    total_energy = energy.sum()
    
    if total_energy < 1e-10:
        # Degenerate case
        return torch.ones_like(energy)
    
    cum_energy = torch.cumsum(energy, dim=0)
    return cum_energy / total_energy


def select_rank(
    singular_values: torch.Tensor,
    energy_threshold: float = 0.90,
    max_rank: Optional[int] = None,
    min_rank: int = 1
) -> int:
    """
    Select rank based on cumulative energy threshold.
    
    Args:
        singular_values: 1D tensor of singular values
        energy_threshold: Retain this fraction of energy
        max_rank: Maximum rank cap (None for no cap)
        min_rank: Minimum rank
        
    Returns:
        Selected rank k
    """
    cum_energy = compute_energy_spectrum(singular_values)
    
    # Find first index where cumulative energy >= threshold
    k = int((cum_energy < energy_threshold).sum().item()) + 1
    
    # Apply constraints
    k = max(k, min_rank)
    if max_rank is not None:
        k = min(k, max_rank)
    
    # Can't exceed number of singular values
    k = min(k, len(singular_values))
    
    return k


def compute_svd(
    matrix: torch.Tensor,
    full_matrices: bool = False,
    use_randomized: bool = False,
    random_rank: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD with optional randomized approximation.
    
    Args:
        matrix: Input matrix [D x N]
        full_matrices: Whether to compute full U and V matrices
        use_randomized: Use randomized SVD approximation
        random_rank: Target rank for randomized SVD
        
    Returns:
        Tuple of (U, S, Vh) where U is [D x k], S is [k], Vh is [k x N]
    """
    if use_randomized and random_rank is not None:
        # Use randomized SVD (not implemented here, fallback to torch)
        # In production, could use sklearn.utils.extmath.randomized_svd
        print(f"Warning: Randomized SVD not implemented, using standard SVD")
    
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, Vh
    except RuntimeError as e:
        print(f"SVD failed on CUDA, trying CPU: {e}")
        # Fallback to CPU
        matrix_cpu = matrix.cpu()
        U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=full_matrices)
        return U, S, Vh


def construct_basis(
    deltas: List[torch.Tensor],
    energy_threshold: float = 0.90,
    max_rank: Optional[int] = None,
    center: bool = True,
    device: str = "cpu",
    use_randomized: bool = False
) -> Dict:
    """
    Construct SVD basis from task delta vectors.
    
    Args:
        deltas: List of 1D delta tensors
        energy_threshold: Energy retention threshold
        max_rank: Maximum rank cap
        center: Whether to center the matrix
        device: Device for computation
        use_randomized: Use randomized SVD
        
    Returns:
        Dictionary containing:
            - U_high: High-energy basis [D x k]
            - U_low: Low-energy basis [D x (D-k)]
            - singular_values: Singular values
            - k: Selected rank
            - mean: Mean vector (if centered)
            - energy_retained: Fraction of energy retained
            - D: Dimension
            - N: Number of tasks
    """
    if not deltas:
        raise ValueError("Empty delta list")
    
    # Stack and center
    T, mean = stack_and_center(deltas, center)
    T = T.to(device)
    
    D, N = T.shape
    
    # Compute SVD
    U, S, Vh = compute_svd(T, full_matrices=False, use_randomized=use_randomized)
    
    # Select rank
    k = select_rank(S, energy_threshold, max_rank)
    
    # Split into high and low energy bases
    U_high = U[:, :k].contiguous()
    U_low = U[:, k:].contiguous()
    
    # Compute energy retained
    energy_retained = compute_energy_spectrum(S)[k-1].item()
    
    result = {
        "U_high": U_high,
        "U_low": U_low,
        "singular_values": S,
        "k": k,
        "mean": mean,
        "energy_retained": energy_retained,
        "D": D,
        "N": N
    }
    
    return result


def construct_masked_basis(
    masked_deltas: List[torch.Tensor],
    unmasked_deltas: Optional[List[torch.Tensor]],
    energy_threshold: float = 0.90,
    max_rank: Optional[int] = None,
    center: bool = True,
    device: str = "cpu",
    include_noise: bool = False
) -> Dict:
    """
    Construct basis for masked (signal) and optionally unmasked (noise) regions.
    
    Args:
        masked_deltas: List of masked delta vectors
        unmasked_deltas: List of unmasked delta vectors (optional)
        energy_threshold: Energy retention threshold
        max_rank: Maximum rank cap
        center: Whether to center
        device: Device for computation
        include_noise: Whether to construct noise basis
        
    Returns:
        Dictionary containing masked basis and optionally noise basis
    """
    result = {}
    
    # Construct masked (signal) basis
    if masked_deltas and len(masked_deltas[0]) > 0:
        masked_basis = construct_basis(
            masked_deltas,
            energy_threshold=energy_threshold,
            max_rank=max_rank,
            center=center,
            device=device
        )
        result["masked"] = masked_basis
    else:
        result["masked"] = None
    
    # Construct unmasked (noise) basis if requested
    if include_noise and unmasked_deltas and len(unmasked_deltas[0]) > 0:
        noise_basis = construct_basis(
            unmasked_deltas,
            energy_threshold=energy_threshold,
            max_rank=max_rank,
            center=center,
            device=device
        )
        result["noise"] = noise_basis
    else:
        result["noise"] = None
    
    return result


def compute_energy_statistics(singular_values: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about energy distribution.
    
    Args:
        singular_values: 1D tensor of singular values
        
    Returns:
        Dictionary of statistics
    """
    energy = singular_values ** 2
    total_energy = energy.sum().item()
    
    if len(singular_values) > 0:
        top_energy = energy[0].item()
        top_energy_ratio = top_energy / total_energy if total_energy > 0 else 0
    else:
        top_energy = 0
        top_energy_ratio = 0
    
    return {
        "total_energy": total_energy,
        "top_singular_value": singular_values[0].item() if len(singular_values) > 0 else 0,
        "top_energy_ratio": top_energy_ratio,
        "num_components": len(singular_values)
    }
