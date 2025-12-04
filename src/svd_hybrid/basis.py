"""
SVD basis construction utilities with energy-based rank selection.

=== TUTORIAL: Understanding SVD for Task Vector Merging ===

Singular Value Decomposition (SVD) is a fundamental matrix factorization technique
that decomposes any matrix into three components:

    M = U × Σ × V^T

Where:
- U: Left singular vectors (column space basis)
- Σ: Diagonal matrix of singular values (importance weights)
- V^T: Right singular vectors (row space basis)

=== WHY USE SVD FOR TASK VECTORS? ===

When merging multiple task vectors, we stack them into a matrix where:
- Each column is one task's delta vector
- Each row is one parameter dimension

SVD finds the "principal directions" that best explain the variation across tasks:
- **High singular values** = Important directions shared across tasks
- **Low singular values** = Less important or task-specific variations

By keeping only high-energy components, we:
1. **Compress** the representation (fewer coefficients to store)
2. **Denoise** by removing low-variance components
3. **Find commonality** across tasks

=== ENERGY-BASED RANK SELECTION ===

Instead of keeping a fixed number of components, we use "energy" (variance):

    energy_i = σ_i² / Σ(σ_j²)

We keep the smallest rank k such that:
    
    cumulative_energy(k) = (σ_1² + σ_2² + ... + σ_k²) / total_energy >= threshold

This adapts to each parameter's structure automatically.

=== EXAMPLE ===

    >>> # Stack task deltas into matrix [D x N] where D=dims, N=num_tasks
    >>> deltas = [task1_delta, task2_delta, task3_delta, task4_delta]  # Each is 1D
    >>> 
    >>> # Construct basis with 95% energy retention
    >>> result = construct_basis(deltas, energy_threshold=0.95, max_rank=64)
    >>> 
    >>> # Result contains:
    >>> # - U_high: High-energy basis [D x k] for storing in FP16
    >>> # - U_low: Low-energy basis [D x (D-k)] for quantization
    >>> # - k: Selected rank
    >>> # - energy_retained: Actual energy fraction retained
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
    
    Takes multiple 1D delta vectors (one per task) and stacks them into a
    2D matrix suitable for SVD. Optionally subtracts the mean across tasks
    to center the data, which often improves SVD decomposition.
    
    === WHY CENTER? ===
    
    Centering removes the "average task vector" from each task's delta,
    making SVD focus on the differences between tasks rather than their
    common direction. This often results in:
    - More meaningful principal components
    - Better separation of task-specific variations
    - Improved reconstruction quality
    
    Args:
        vectors: List of 1D tensors (all same length D), one per task
        center: Whether to mean-center columns (subtract mean across tasks)
        
    Returns:
        Tuple of:
            - T: Stacked matrix [D x N] where N = number of tasks
            - mean: Mean vector [D x 1] if centered, None otherwise
            
    Example:
        >>> deltas = [torch.randn(100) for _ in range(4)]  # 4 tasks, 100 dims
        >>> T, mean = stack_and_center(deltas, center=True)
        >>> T.shape
        torch.Size([100, 4])
    """
    if not vectors:
        raise ValueError("Empty vector list")
    
    # Stack into matrix: each vector becomes a column
    # Result shape: [D x N] where D = dimension, N = number of tasks
    T = torch.stack(vectors, dim=1)
    
    # Optionally center by subtracting mean across columns (tasks)
    mean = None
    if center:
        # Compute mean vector [D x 1]
        mean = T.mean(dim=1, keepdim=True)
        # Subtract mean from each column
        T = T - mean
    
    return T, mean


def compute_energy_spectrum(singular_values: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative energy spectrum from singular values.
    
    "Energy" in the SVD context refers to the squared singular values,
    which represent the variance explained by each principal component.
    
    The cumulative energy at index k tells us what fraction of total
    variance is captured by the first k+1 components.
    
    === FORMULA ===
    
    energy_i = σ_i²
    cumulative_energy(k) = Σ(σ_1² + ... + σ_k²) / Σ(all σ_i²)
    
    Args:
        singular_values: 1D tensor of singular values (sorted descending by convention)
        
    Returns:
        Tensor of cumulative energy fractions, same length as input
        Values range from small (first component alone) to 1.0 (all components)
        
    Example:
        >>> S = torch.tensor([10.0, 5.0, 2.0, 1.0])  # Singular values
        >>> cum_energy = compute_energy_spectrum(S)
        >>> # Energy: [100, 25, 4, 1] -> Total = 130
        >>> # Cumulative: [100/130, 125/130, 129/130, 130/130]
        >>> cum_energy[0]  # ~0.77 (first component captures 77%)
        >>> cum_energy[-1]  # 1.0 (all components capture 100%)
    """
    # Square singular values to get variance (energy)
    energy = singular_values ** 2
    total_energy = energy.sum()
    
    # Handle degenerate case where all singular values are ~0
    if total_energy < 1e-10:
        return torch.ones_like(energy)
    
    # Compute cumulative sum and normalize
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
    
    Finds the smallest rank k such that the cumulative energy (variance)
    captured by the first k singular values meets or exceeds the threshold.
    
    === ALGORITHM ===
    
    1. Compute cumulative energy: E_k = (σ_1² + ... + σ_k²) / total
    2. Find smallest k where E_k >= threshold
    3. Apply min_rank and max_rank constraints
    
    === TRADEOFFS ===
    
    - Higher threshold (e.g., 0.99): More accuracy, less compression
    - Lower threshold (e.g., 0.80): More compression, potential quality loss
    - max_rank cap: Prevents excessive rank even for complex parameters
    - min_rank: Ensures at least some representation even for simple data
    
    Args:
        singular_values: 1D tensor of singular values
        energy_threshold: Target fraction of energy to retain (0.0-1.0)
        max_rank: Maximum rank cap (None = no cap)
        min_rank: Minimum rank (default: 1)
        
    Returns:
        Selected rank k (integer)
        
    Example:
        >>> S = torch.tensor([10.0, 5.0, 2.0, 1.0])
        >>> k = select_rank(S, energy_threshold=0.90)  # First 2 components
        >>> k = select_rank(S, energy_threshold=0.95, max_rank=2)  # Capped at 2
    """
    # Get cumulative energy spectrum
    cum_energy = compute_energy_spectrum(singular_values)
    
    # Find first index where cumulative energy >= threshold
    # Count how many are BELOW threshold, then add 1
    k = int((cum_energy < energy_threshold).sum().item()) + 1
    
    # Apply minimum and maximum rank constraints
    k = max(k, min_rank)
    if max_rank is not None:
        k = min(k, max_rank)
    
    # Can't exceed number of available singular values
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
    
    original_device = matrix.device
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, Vh
    except RuntimeError as e:
        print(f"SVD failed on CUDA, trying CPU: {e}")
        # Fallback to CPU
        matrix_cpu = matrix.cpu()
        U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=full_matrices)
        # Move results back to the original device
        return U.to(original_device), S.to(original_device), Vh.to(original_device)


def construct_basis(
    deltas: List[torch.Tensor],
    energy_threshold: float = 0.90,
    max_rank: Optional[int] = None,
    center: bool = True,
    device: str = "cpu",
    use_randomized: bool = False,
    verbose: bool = True
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
        verbose: Print tutorial-style progress messages
        
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
    
    if verbose:
        print(f"\n{'─'*60}")
        print(f"🔬 SVD Basis Construction")
        print(f"{'─'*60}")
        print(f"""
   🎯 WHAT IS SVD DOING HERE?
      SVD (Singular Value Decomposition) finds the "principal directions"
      that best explain how task vectors vary. Think of it like finding
      the main axes of variation in a cloud of data points.
      
      M = U × Σ × V^T
      
      Where:
      • U contains the basis vectors (directions in parameter space)
      • Σ contains singular values (importance of each direction)
      • We keep high-Σ directions (important) in high precision
      • We quantize low-Σ directions (less important) to save space
""")
    
    # Stack and center
    T, mean = stack_and_center(deltas, center)
    T = T.to(device)
    
    D, N = T.shape
    
    if verbose:
        print(f"   📊 INPUT DATA:")
        print(f"      └─ Number of tasks (N): {N}")
        print(f"      └─ Parameter dimension (D): {D:,}")
        print(f"      └─ Matrix shape: {D} × {N}")
        print(f"      └─ Centering: {'Yes (subtracting mean)' if center else 'No'}")
        if center and mean is not None:
            mean_norm = mean.norm().item()
            print(f"      └─ Mean vector norm: {mean_norm:.6f}")
    
    # Compute SVD
    if verbose:
        print(f"\n   🔄 Computing SVD decomposition...")
    
    U, S, Vh = compute_svd(T, full_matrices=False, use_randomized=use_randomized)
    
    # Compute energy spectrum
    sv = S.cpu().numpy()
    energy = sv**2
    total_energy = energy.sum()
    cum_energy = energy.cumsum() / total_energy if total_energy > 0 else np.ones_like(energy)
    
    if verbose:
        print(f"   ✅ SVD completed!")
        print(f"\n   📊 SINGULAR VALUE ANALYSIS:")
        print(f"      └─ Number of singular values: {len(S)}")
        print(f"      └─ Largest singular value: {sv[0]:.6f}")
        print(f"      └─ Smallest singular value: {sv[-1]:.6f}" if len(sv) > 1 else "")
        print(f"      └─ Total energy (sum of σ²): {total_energy:.6f}")
        
        # Show energy distribution
        print(f"\n   📈 CUMULATIVE ENERGY SPECTRUM:")
        print(f"      (How much variance is captured by first k components)")
        milestones = [0.5, 0.75, 0.9, 0.95, 0.99]
        for milestone in milestones:
            k_needed = int((cum_energy < milestone).sum()) + 1
            if k_needed <= len(cum_energy):
                print(f"      └─ {milestone*100:5.1f}% energy: k = {k_needed}")
    
    # Select rank
    k = select_rank(S, energy_threshold, max_rank)
    
    if verbose:
        print(f"\n   🎯 RANK SELECTION:")
        print(f"      └─ Energy threshold: {energy_threshold*100:.1f}%")
        print(f"      └─ Max rank cap: {max_rank if max_rank else 'None'}")
        print(f"      └─ Selected rank k: {k}")
        actual_energy = cum_energy[k-1] if k > 0 else 0
        print(f"      └─ Actual energy retained: {actual_energy*100:.2f}%")
    
    # Split into high and low energy bases
    U_high = U[:, :k].contiguous()
    U_low = U[:, k:].contiguous()
    
    # Compute energy retained
    energy_retained = compute_energy_spectrum(S)[k-1].item() if k > 0 else 0
    
    if verbose:
        print(f"\n   📦 BASIS SPLIT:")
        print(f"      └─ U_high shape: [{D}, {k}] (high-energy, stored in FP16)")
        print(f"      └─ U_low shape: [{D}, {U_low.shape[1]}] (low-energy, will be quantized)")
        
        # Validation
        print(f"\n   🔬 VALIDATION CHECKS:")
        
        # Check 1: Energy threshold met
        if energy_retained >= energy_threshold:
            print(f"   ✅ CHECK 1 PASSED: Energy threshold met ({energy_retained*100:.2f}% >= {energy_threshold*100:.1f}%)")
        else:
            print(f"   ⚠️ CHECK 1 WARNING: Energy below threshold ({energy_retained*100:.2f}% < {energy_threshold*100:.1f}%)")
            print(f"      └─ This can happen when max_rank limits the rank selection")
        
        # Check 2: Basis is orthogonal (should be by SVD construction)
        if U_high.shape[1] > 0:
            orthogonality = (U_high.T @ U_high - torch.eye(k, device=U_high.device)).abs().max().item()
            if orthogonality < 1e-5:
                print(f"   ✅ CHECK 2 PASSED: Basis is orthogonal (error={orthogonality:.2e})")
            else:
                print(f"   ⚠️ CHECK 2 WARNING: Orthogonality error={orthogonality:.2e}")
        
        # Check 3: Reasonable compression
        compression = D / k if k > 0 else float('inf')
        print(f"   ✅ CHECK 3: Dimension reduction {D} → {k} ({compression:.1f}x)")
        
        print(f"{'─'*60}\n")
    
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
    include_noise: bool = False,
    verbose: bool = False
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
        verbose: Print progress messages (passed to construct_basis)
        
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
            device=device,
            verbose=verbose
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
            device=device,
            verbose=verbose
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
