"""
Command-line interface for SVD-Hybrid merging.

=== TUTORIAL: Using the CLI ===

This module provides the command-line interface for running SVD-Hybrid merging.
It supports configuration via command-line arguments and JSON config files.

=== BASIC USAGE ===

    python -m src.svd_hybrid.cli \\
        --tasks Cars DTD EuroSAT SUN397 \\
        --checkpoint-dir ./checkpoints \\
        --base-model-path ./base_model.pt \\
        --output-dir ./output

=== CONFIGURATION OPTIONS ===

The CLI accepts three types of configuration:

1. **Command-line arguments**: Direct control
   --energy-threshold 0.95
   --max-rank 64
   --weighting performance

2. **Main config file** (--config): JSON with all settings
   {"tasks": [...], "svd_energy_threshold": 0.95}

3. **Specialized configs**:
   --quantize-config: Quantization settings
   --load-config: Loading settings (for quantized inputs)

=== PIPELINE STEPS ===

The CLI runs through 9 steps:
1. Load task vectors from checkpoints
2. Load and combine masks (if provided)
3. Extract parameter information
4. Construct SVD bases
5. Compress task vectors
6. Compute task weights
7. Merge parameters
8. Create merged model
9. Compute diagnostics

=== OUTPUT FILES ===

After running, you'll find:
- output_dir/merged_state_dict.pt: The merged model
- output_dir/weights.json: Task weights used
- output_dir/clusters.json: Cluster assignments (if clustering)
- artifact_dir/: Bases, coefficients, diagnostics (if --store-artifacts)
"""

import argparse
import torch
import os
from typing import Optional, Dict
from .config import SVDHybridConfig
from .task_vector_loader import load_task_vectors, get_task_checkpoint_paths, get_parameter_names
from .mask_loader import load_task_masks, combine_masks
from .basis import construct_masked_basis
from .compress import compress_all_parameters
from .merge import merge_all_parameters, apply_merged_deltas, merge_with_clustering
from .storage import save_all_artifacts, save_merged_model
from .diagnostics import compute_all_diagnostics, print_diagnostics_summary, compute_compression_statistics, print_detailed_compression_report
from .weighting import compute_weights
from .clustering import cluster_tasks
from .mask_loader import apply_mask_to_tensor, get_unmasked_portion
from .task_vector_loader import load_checkpoint


def run_svd_hybrid_pipeline(config: SVDHybridConfig) -> Dict:
    """
    Run the complete SVD-Hybrid merging pipeline.
    
    This is the main entry point for programmatic use. It executes all
    9 steps of the SVD-Hybrid method:
    
    1. Load task vectors from checkpoints
    2. Load and combine masks
    3. Extract parameter information
    4. Construct SVD bases for each parameter
    5. Compress task vectors (project + quantize)
    6. Compute task weights
    7. Merge parameters (dequantize + average + reconstruct)
    8. Create merged model (base + deltas)
    9. Compute diagnostics and save results
    
    Args:
        config: SVDHybridConfig object with all parameters
        
    Returns:
        Dictionary containing:
            - merged_state_dict: The merged model state dict
            - diagnostics: Reconstruction errors and metrics
            - bases: SVD bases for all parameters
            - compressed: Compressed coefficients for all tasks
            
    Example:
        >>> config = SVDHybridConfig(
        ...     tasks=["Cars", "DTD"],
        ...     checkpoint_dir="./ckpts",
        ...     base_model_path="./base.pt"
        ... )
        >>> results = run_svd_hybrid_pipeline(config)
        >>> model.load_state_dict(results["merged_state_dict"])
    """
    # Print tutorial header
    print(f"\n{'='*80}")
    print(f"📚 SVD-HYBRID MODEL MERGING PIPELINE")
    print(f"{'='*80}")
    print(f"""
🎯 WHAT IS SVD-HYBRID?
   SVD-Hybrid is an advanced method for merging multiple fine-tuned models into one.
   It combines:
   • SVD (Singular Value Decomposition) - finds the most important directions
   • Quantization - compresses the model changes efficiently
   • Task Arithmetic - combines what different models learned
   
   The result: A single model that can perform multiple tasks!

📋 PIPELINE OVERVIEW (9 Steps):
   1. Load task vectors (what each model learned)
   2. Load and combine masks (which parameters matter)
   3. Extract parameter information
   4. Construct SVD bases (find important directions)
   5. Compress task vectors (reduce storage)
   6. Compute task weights (how much each task matters)
   7. Merge parameters (combine the learning)
   8. Create merged model
   9. Compute diagnostics (verify quality)
""")
    
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    print(f"📁 Tasks to merge: {config.tasks}")
    print(f"{'='*80}\n")
    
    # Load base model state dict
    print(f"{'─'*80}")
    print(f"📂 PRELIMINARY: Loading Base Model")
    print(f"{'─'*80}")
    print(f"   └─ The base model is the starting point before any task-specific training")
    print(f"   └─ Path: {config.base_model_path}")
    base_state_dict = load_checkpoint(config.base_model_path, device=device)
    num_base_params = len(base_state_dict)
    total_base_elements = sum(p.numel() for p in base_state_dict.values() if hasattr(p, 'numel'))
    print(f"   ✅ LOADED: {num_base_params:,} parameters, {total_base_elements:,} total elements")
    
    # =========================================================================
    # Step 1: Load task vectors
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 1/9: Loading Task Vectors")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHAT ARE TASK VECTORS?
      Each task vector represents what a model "learned" during fine-tuning:
      task_vector = fine_tuned_weights - base_weights
      
      By loading these, we can see exactly what changed for each task.
""")
    task_checkpoint_paths = get_task_checkpoint_paths(config.checkpoint_dir, config.tasks)
    print(f"   📁 Checkpoint directory: {config.checkpoint_dir}")
    print(f"   📋 Tasks to load: {config.tasks}")
    
    task_vectors = load_task_vectors(
        config.base_model_path,
        task_checkpoint_paths,
        device=device
    )
    
    # Validation
    print(f"\n   🔬 VALIDATION:")
    if len(task_vectors) == len(config.tasks):
        print(f"   ✅ CHECK PASSED: All {len(task_vectors)} task vectors loaded successfully")
    else:
        print(f"   ⚠️ WARNING: Expected {len(config.tasks)} tasks, got {len(task_vectors)}")
    
    # Show stats per task
    print(f"\n   📊 TASK VECTOR STATISTICS:")
    for task_name, tv in task_vectors.items():
        total_norm = sum(d.norm().item()**2 for d in tv.values()) ** 0.5
        print(f"      └─ {task_name}: {len(tv)} params, total change norm: {total_norm:.4f}")
    
    # =========================================================================
    # Step 2: Load and combine masks
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 2/9: Loading and Combining Masks")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHAT ARE MASKS?
      Masks identify which parameters are "important" for each task.
      By combining masks, we focus on parameters that matter across tasks.
      
      Strategy '{config.svd_mask_strategy}' will be used to combine them.
""")
    
    if config.mask_dir and os.path.exists(config.mask_dir):
        print(f"   📁 Mask directory: {config.mask_dir}")
        task_masks = load_task_masks(
            config.mask_dir, 
            config.tasks, 
            device=device,
            reference_state_dict=base_state_dict
        )
        combined_masks = combine_masks(task_masks, strategy=config.svd_mask_strategy, device=device)
        
        # Validation
        if len(combined_masks) > 0:
            total_masked = sum(m.sum().item() for m in combined_masks.values())
            total_elements = sum(m.numel() for m in combined_masks.values())
            mask_ratio = total_masked / max(total_elements, 1) * 100
            print(f"   ✅ LOADED: {len(combined_masks)} parameter masks")
            print(f"   📊 Coverage: {mask_ratio:.1f}% of parameters are masked (considered important)")
        else:
            print(f"   ⚠️ WARNING: No masks found or loaded")
    else:
        print(f"   ℹ️ No mask directory provided - using all parameters")
        print(f"   └─ This means all parameters will be considered equally")
        combined_masks = {}
    
    # =========================================================================
    # Step 3: Get parameter names and shapes
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 3/9: Extracting Parameter Information")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHY THIS STEP?
      We need to know the structure of all parameters to process them correctly.
      This ensures we handle each layer/weight matrix appropriately.
""")
    
    param_names = get_parameter_names(task_vectors)
    original_shapes = {}
    total_params = 0
    for task_vector in task_vectors.values():
        for param_name, delta in task_vector.items():
            if param_name not in original_shapes:
                original_shapes[param_name] = delta.shape
                total_params += delta.numel()
    
    print(f"   📊 Found {len(param_names):,} unique parameters")
    print(f"   📊 Total parameter elements: {total_params:,}")
    
    # Show sample of parameter names
    sample_params = list(param_names)[:5]
    print(f"   📋 Sample parameters: {sample_params}")
    if len(param_names) > 5:
        print(f"      ... and {len(param_names) - 5} more")
    
    print(f"   ✅ VALIDATION: Parameter info extracted successfully")
    
    # =========================================================================
    # Step 4: Construct SVD bases for each parameter
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 4/9: Constructing SVD Bases")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHAT IS SVD?
      SVD (Singular Value Decomposition) finds the "principal directions" of change.
      
      Think of it like this: if multiple tasks all change weights in similar ways,
      SVD finds those common patterns. This lets us:
      • Compress the representation (fewer numbers to store)
      • Focus on what matters (high-energy = important changes)
      
   ⚙️ Settings:
      • Energy threshold: {config.svd_energy_threshold} (keep {config.svd_energy_threshold*100}% of variance)
      • Max rank: {config.svd_max_rank} (limit complexity)
      • Center data: {config.svd_center}
""")
    
    # Print mathematical overview for SVD
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SVD MATHEMATICAL PROCESS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 FOR EACH PARAMETER (weight matrix/bias vector):                         │
│                                                                              │
│  Step 1: Stack task deltas into matrix T                                    │
│          T = [δ₁ | δ₂ | ... | δₙ] ∈ ℝ^(D×N)                                │
│          where D = flattened parameter dimension                            │
│                N = number of tasks ({len(config.tasks)})                                 │
│                                                                              │
│  Step 2: Center the data (optional, enabled: {config.svd_center})                         │
│          T_centered = T - mean(T, axis=1)                                   │
│                                                                              │
│  Step 3: Compute SVD decomposition                                          │
│          T = U × Σ × Vᵀ                                                     │
│          • U ∈ ℝ^(D×D): Left singular vectors (basis directions)           │
│          • Σ = diag(σ₁, σ₂, ..., σₘ): Singular values (importance)         │
│          • Vᵀ ∈ ℝ^(N×N): Right singular vectors                            │
│                                                                              │
│  Step 4: Select rank k based on energy threshold ({config.svd_energy_threshold})               │
│          Find smallest k such that:                                         │
│          (σ₁² + σ₂² + ... + σₖ²) / (σ₁² + σ₂² + ... + σₘ²) ≥ {config.svd_energy_threshold}         │
│                                                                              │
│  Step 5: Split basis                                                        │
│          U_high = U[:, :k]   → Top k directions (high energy)              │
│          U_low = U[:, k:]    → Remaining directions (low energy)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    bases = {}
    successful_bases = 0
    total_energy_retained = 0
    ranks_selected = []
    dimensions = []
    
    for i, param_name in enumerate(param_names):
        # Extract masked and unmasked deltas
        mask = combined_masks.get(param_name)
        
        masked_deltas = []
        unmasked_deltas = []
        
        for task_name, task_vector in task_vectors.items():
            if param_name not in task_vector:
                continue
            
            delta = task_vector[param_name]
            
            if mask is not None and mask.shape == delta.shape:
                # Apply mask
                if mask.sum() >= config.svd_min_mask_size:
                    masked = apply_mask_to_tensor(delta, mask)
                    masked_deltas.append(masked)
                    
                    if config.svd_include_noise:
                        unmasked = get_unmasked_portion(delta, mask)
                        unmasked_deltas.append(unmasked)
            else:
                # No mask, use entire parameter
                masked_deltas.append(delta.flatten())
        
        if masked_deltas and len(masked_deltas[0]) > 0:
            basis = construct_masked_basis(
                masked_deltas,
                unmasked_deltas if config.svd_include_noise else None,
                energy_threshold=config.svd_energy_threshold,
                max_rank=config.svd_max_rank,
                center=config.svd_center,
                device=device,
                include_noise=config.svd_include_noise
            )
            
            # Convert to FP16 if requested
            if config.svd_fp16 and basis.get("masked") is not None:
                basis["masked"]["U_high"] = basis["masked"]["U_high"].half()
                basis["masked"]["U_low"] = basis["masked"]["U_low"].half()
                
                if basis.get("noise") is not None:
                    basis["noise"]["U_high"] = basis["noise"]["U_high"].half()
                    basis["noise"]["U_low"] = basis["noise"]["U_low"].half()
            
            bases[param_name] = basis
            
            if basis.get("masked"):
                successful_bases += 1
                total_energy_retained += basis['masked']['energy_retained']
                ranks_selected.append(basis['masked']['k'])
                dimensions.append(basis['masked']['D'])
                
                # Print progress for significant parameters with more detail
                if i < 5 or i % 50 == 0:
                    k = basis['masked']['k']
                    D = basis['masked']['D']
                    energy = basis['masked']['energy_retained']
                    reduction = D / k if k > 0 else float('inf')
                    print(f"   ├─ {param_name[:35]}{'...' if len(param_name) > 35 else '':<5}")
                    print(f"   │     └─ D={D:,} → k={k} ({reduction:.1f}x dim reduction), energy={energy:.4f}")
    
    # Validation and summary
    avg_energy = total_energy_retained / max(successful_bases, 1)
    avg_rank = sum(ranks_selected) / max(len(ranks_selected), 1)
    avg_dim = sum(dimensions) / max(len(dimensions), 1)
    total_original_dim = sum(dimensions)
    total_compressed_dim = sum(ranks_selected)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SVD BASIS CONSTRUCTION SUMMARY                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📊 CONSTRUCTION RESULTS:                                                   │
│     Bases constructed: {successful_bases}/{len(param_names)} parameters                              │
│     Total dimension (Σ D): {total_original_dim:,} elements                              │
│     Total rank (Σ k): {total_compressed_dim:,} elements                                  │
│     Dimensionality reduction: {total_original_dim/max(total_compressed_dim,1):.1f}x                                     │
│                                                                              │
│  📈 RANK STATISTICS:                                                        │
│     Average rank k: {avg_rank:.1f}                                              │
│     Rank range: [{min(ranks_selected) if ranks_selected else 0}, {max(ranks_selected) if ranks_selected else 0}]                                                │
│     Average dimension D: {avg_dim:.1f}                                          │
│                                                                              │
│  ⚡ ENERGY RETENTION:                                                        │
│     Target threshold: {config.svd_energy_threshold*100:.1f}%                                           │
│     Achieved average: {avg_energy*100:.2f}%                                          │
│     Status: {'✅ Target met' if avg_energy >= config.svd_energy_threshold - 0.05 else '⚠️ Below target'}                                              │
│                                                                              │
│  💾 BASIS STORAGE FORMAT:                                                    │
│     U_high: {'FP16' if config.svd_fp16 else 'FP32'} (2 bytes/value) - High energy directions         │
│     U_low: {'FP16' if config.svd_fp16 else 'FP32'} (2 bytes/value) - Low energy directions          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    # =========================================================================
    # Step 5: Compress task vectors
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 5/9: Compressing Task Vectors")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHY COMPRESS?
      After SVD, we have coefficients for each task. Now we quantize (compress)
      the less important coefficients to save space.
      
   ⚙️ Compression Settings:
      • Low-energy bits: {config.svd_low_bits} bits
      • RTVQ stages: {config.svd_rtvq_stages} (more stages = better quality)
      
   💡 How it works:
      • High-energy coefficients → FP16 (preserve accuracy)
      • Low-energy coefficients → {config.svd_low_bits}-bit RTVQ (save space)
""")
    
    compressed_all = compress_all_parameters(
        task_vectors,
        combined_masks,
        bases,
        config,
        device=device
    )
    
    print(f"   ✅ COMPRESSED: {len(compressed_all)} parameters")
    
    # Compute and print detailed compression statistics
    compression_stats = compute_compression_statistics(
        task_vectors,
        compressed_all,
        bases,
        config
    )
    print_detailed_compression_report(compression_stats, config)
    
    # =========================================================================
    # Step 6: Compute task weights
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 6/9: Computing Task Weights")
    print(f"{'─'*80}")
    print(f"""
   🎯 WHY WEIGHT TASKS?
      Not all tasks are equally important or perform equally well.
      Weighting lets us give more influence to better-performing tasks.
      
   ⚙️ Weighting Strategy: '{config.svd_weighting}'
""")
    
    # Print mathematical explanation based on weighting strategy
    if config.svd_weighting == "uniform":
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNIFORM WEIGHTING FORMULA                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 FORMULA:                                                                │
│     w_i = 1/N  for all tasks i ∈ {{1, 2, ..., N}}                           │
│                                                                              │
│     where N = {len(config.tasks)} (number of tasks)                                        │
│           w_i = 1/{len(config.tasks)} = {1.0/len(config.tasks):.4f}                                                  │
│                                                                              │
│  📝 Each task contributes equally to the merged model                       │
│     Constraint: Σw_i = 1                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    elif config.svd_weighting == "performance":
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PERFORMANCE-BASED WEIGHTING FORMULA                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 FORMULA (softmax with temperature):                                     │
│     w_i = exp(p_i / τ) / Σⱼ exp(p_j / τ)                                    │
│                                                                              │
│     where p_i = performance (accuracy) of task i                            │
│           τ = temperature = {config.svd_weighting_temperature} (higher = more uniform)                 │
│                                                                              │
│  📝 Tasks with higher accuracy get more weight                              │
│     Temperature controls how much we favor high performers                  │
│     τ → 0: winner-take-all, τ → ∞: uniform weights                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    elif config.svd_weighting == "cluster":
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLUSTER-BASED WEIGHTING FORMULA                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 TWO-STAGE WEIGHTING:                                                    │
│                                                                              │
│  Stage 1: Within-cluster averaging                                          │
│     c_k = Σᵢ∈cluster_k (w_i × δ_i) / |cluster_k|                           │
│                                                                              │
│  Stage 2: Across-cluster averaging                                          │
│     merged = Σₖ w_cluster_k × c_k                                           │
│                                                                              │
│  📝 Similar tasks are grouped and averaged first                            │
│     Then cluster results are combined                                       │
│     Number of clusters: {config.svd_cluster_k}                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    cluster_assignments = None
    
    if config.svd_weighting == "cluster":
        print(f"   🔄 Clustering tasks into {config.svd_cluster_k} groups...")
        cluster_assignments = cluster_tasks(task_vectors, config.svd_cluster_k, method="kmeans")
        
        from .clustering import get_cluster_members
        clusters = get_cluster_members(cluster_assignments)
        for cluster_id, members in clusters.items():
            print(f"      └─ Cluster {cluster_id}: {members}")
    
    weights = compute_weights(
        config.tasks,
        weighting_strategy=config.svd_weighting,
        performance_file=config.performance_file,
        temperature=config.svd_weighting_temperature,
        cluster_assignments=cluster_assignments
    )
    
    print(f"\n   📊 TASK WEIGHTS:")
    total_weight = 0
    for task_name, weight in sorted(weights.items()):
        bar_len = int(weight * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"      {task_name:15s}: {weight:.4f} [{bar}]")
        total_weight += weight
    
    # Validation
    if abs(total_weight - 1.0) < 0.01:
        print(f"   ✅ VALIDATION: Weights sum to {total_weight:.4f} (should be ~1.0)")
    else:
        print(f"   ⚠️ WARNING: Weights sum to {total_weight:.4f} (expected ~1.0)")
    
    # =========================================================================
    # Step 7: Merge parameters
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 7/9: Merging Parameters")
    print(f"{'─'*80}")
    print(f"""
   🎯 THE MAGIC STEP!
      Now we combine all the task vectors into one merged representation.
      
   💡 Process:
      1. Dequantize the compressed coefficients
      2. Compute weighted average: merged = Σ(weight_i × task_i)
      3. Reconstruct the parameter deltas from coefficients
""")
    
    # Print mathematical explanation for merging
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MERGING MATHEMATICAL PROCESS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 FOR EACH PARAMETER, MERGING HAPPENS IN COEFFICIENT SPACE:               │
│                                                                              │
│  Step 1: DEQUANTIZE low-energy coefficients                                 │
│          c_low = dequant(q₁) + dequant(q₂) + ... (RTVQ reconstruction)     │
│                                                                              │
│  Step 2: WEIGHTED AVERAGE of coefficients across tasks                      │
│          c_high_merged = Σᵢ (wᵢ × c_high_i)                                 │
│          c_low_merged = Σᵢ (wᵢ × c_low_i)                                   │
│                                                                              │
│          where wᵢ = weight for task i (Σwᵢ = 1)                             │
│                                                                              │
│  Step 3: RECONSTRUCT merged delta in parameter space                        │
│          δ_merged = U_high × c_high_merged + U_low × c_low_merged          │
│                                                                              │
│          This reverses the SVD projection to get the merged delta          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    if config.svd_weighting == "cluster" and cluster_assignments is not None:
        print(f"   └─ Using cluster-based merging (hierarchical)")
        merged_deltas = merge_with_clustering(
            compressed_all,
            bases,
            combined_masks,
            weights,
            cluster_assignments,
            original_shapes,
            config,
            device=device
        )
    else:
        print(f"   └─ Using direct weighted averaging")
        merged_deltas = merge_all_parameters(
            compressed_all,
            bases,
            combined_masks,
            weights,
            original_shapes,
            config,
            device=device
        )
    
    print(f"   ✅ MERGED: {len(merged_deltas)} parameters")
    
    # Compute merge statistics
    total_merge_norm = sum(d.norm().item()**2 for d in merged_deltas.values()) ** 0.5
    print(f"   📊 Total merged delta norm: {total_merge_norm:.4f}")
    
    # =========================================================================
    # Step 8: Apply to base model
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 8/9: Creating Merged Model")
    print(f"{'─'*80}")
    print(f"""
   🎯 FINAL ASSEMBLY!
      We add the merged deltas back to the base model:
      merged_model = base_model + merged_deltas
      
      This creates a single model that combines all task knowledge!
""")
    
    # Print mathematical explanation
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINAL MODEL ASSEMBLY FORMULA                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  📐 TASK ARITHMETIC FORMULA:                                                │
│     θ_merged = θ_base + α × δ_merged                                        │
│                                                                              │
│     where θ_base = base (pre-trained) model parameters                      │
│           δ_merged = combined task vector from SVD-Hybrid                   │
│           α = scaling factor (typically 1.0)                                │
│                                                                              │
│  📝 The merged model inherits:                                              │
│     • Base knowledge from θ_base                                            │
│     • Combined task-specific adaptations from δ_merged                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Make a copy of base state dict
    merged_state_dict = apply_merged_deltas(
        {k: v.clone() for k, v in base_state_dict.items()}, 
        merged_deltas, 
        device=device
    )
    
    print(f"   ✅ CREATED: Merged model with {len(merged_state_dict):,} parameters")
    
    # Validation: Check shapes match
    shape_match = all(
        merged_state_dict[k].shape == base_state_dict[k].shape
        for k in base_state_dict.keys()
    )
    if shape_match:
        print(f"   ✅ VALIDATION: All parameter shapes match base model")
    else:
        print(f"   ❌ ERROR: Some parameter shapes don't match!")
    
    # =========================================================================
    # Step 9: Compute diagnostics
    # =========================================================================
    print(f"\n{'─'*80}")
    print(f"📚 STEP 9/9: Computing Diagnostics")
    print(f"{'─'*80}")
    print(f"""
   🎯 QUALITY CHECK
      We measure how well we preserved the original task information
      after all the compression and merging.
""")
    
    if config.svd_eval_reconstruction:
        diagnostics = compute_all_diagnostics(
            task_vectors,
            compressed_all,
            bases,
            combined_masks,
            config,
            device=device
        )
        
        # Add weight information
        diagnostics["task_weights"] = weights
        if cluster_assignments:
            diagnostics["cluster_assignments"] = cluster_assignments
        
        print_diagnostics_summary(diagnostics)
        
        # Additional validation
        summary = diagnostics.get("summary", {})
        avg_error = summary.get("average_reconstruction_error", 0)
        
        if avg_error < 0.05:
            print(f"   ✅ QUALITY CHECK PASSED: Reconstruction error {avg_error:.4f} (<5%)")
        elif avg_error < 0.10:
            print(f"   ⚠️ QUALITY CHECK WARNING: Reconstruction error {avg_error:.4f} (5-10%)")
        else:
            print(f"   ❌ QUALITY CONCERN: High reconstruction error {avg_error:.4f} (>10%)")
    else:
        diagnostics = {"task_weights": weights}
        print(f"   ℹ️ Skipping reconstruction evaluation (disabled in config)")
    
    # Save artifacts if requested
    if config.svd_store_artifacts:
        print(f"\n📦 Saving artifacts to {config.artifact_dir}...")
        save_all_artifacts(
            bases,
            compressed_all,
            diagnostics,
            config,
            config.artifact_dir
        )
        print(f"   ✅ Artifacts saved")
    
    # Save merged model
    print(f"\n💾 Saving merged model to {config.output_dir}...")
    save_merged_model(merged_state_dict, config.output_dir)
    
    # Save weights and clusters separately for convenience
    import json
    os.makedirs(config.output_dir, exist_ok=True)
    
    with open(os.path.join(config.output_dir, "weights.json"), 'w') as f:
        json.dump(weights, f, indent=2)
    
    if cluster_assignments:
        with open(os.path.join(config.output_dir, "clusters.json"), 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 SVD-HYBRID MERGING COMPLETE!")
    print(f"{'='*80}")
    print(f"""
   📊 FINAL SUMMARY:
   ├─ Tasks merged: {len(config.tasks)} ({', '.join(config.tasks)})
   ├─ Parameters processed: {len(merged_deltas):,}
   ├─ Average SVD rank: {avg_rank:.1f}
   ├─ Average energy retained: {avg_energy*100:.1f}%
   ├─ Merged model saved to: {config.output_dir}/merged_state_dict.pt
   └─ Weighting: {config.svd_weighting}
   
   💡 NEXT STEPS:
   • Load the merged model: torch.load('{config.output_dir}/merged_state_dict.pt')
   • Run evaluation on each task to verify performance
   • Compare with individual fine-tuned models
""")
    print(f"{'='*80}\n")
    
    return {
        "merged_state_dict": merged_state_dict,
        "diagnostics": diagnostics,
        "bases": bases,
        "compressed": compressed_all
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SVD-Hybrid merging method combining Tall Masks and TVQ"
    )
    
    # Config file options
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file (overrides command-line args)")
    parser.add_argument("--quantize-config", type=str, default=None,
                       help="Path to quantization config JSON")
    parser.add_argument("--load-config", type=str, default=None,
                       help="Path to loading config JSON")
    
    # Task configuration
    parser.add_argument("--tasks", nargs="+",
                       help="List of task identifiers")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                       help="Model identifier (e.g., ViT-B-32)")
    parser.add_argument("--checkpoint-dir", type=str,
                       help="Directory containing task checkpoints")
    parser.add_argument("--base-model-path", type=str,
                       help="Path to base model checkpoint")
    parser.add_argument("--mask-dir", type=str, default="",
                       help="Directory containing tall masks")
    
    # TVQ-specific loading options
    parser.add_argument("--load-tv-type", type=str, default=None,
                       choices=["standard", "quantized", "quantized_finetuned", "quantized_base_and_tv"],
                       help="Type of task vector to load")
    parser.add_argument("--load-task-bits", type=int, default=8,
                       help="Bits for task vector quantization when loading")
    parser.add_argument("--load-base-bits", type=int, default=8,
                       help="Bits for base model quantization when loading")
    
    # SVD parameters
    parser.add_argument("--energy-threshold", type=float, default=0.95,
                       help="Energy retention threshold for rank selection")
    parser.add_argument("--max-rank", type=int, default=64,
                       help="Maximum rank cap")
    parser.add_argument("--center", action="store_true", default=True,
                       help="Center task matrix before SVD")
    parser.add_argument("--no-center", action="store_false", dest="center",
                       help="Don't center task matrix")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use FP16 for bases")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16",
                       help="Use FP32 for bases")
    
    # Quantization parameters
    parser.add_argument("--low-bits", type=int, default=4,
                       help="Bits for low-energy coefficient quantization")
    parser.add_argument("--rtvq-stages", type=int, default=2,
                       help="Number of RTVQ refinement stages")
    
    # Mask parameters
    parser.add_argument("--mask-strategy", type=str, default="union",
                       choices=["union", "intersection", "majority"],
                       help="Mask combination strategy")
    parser.add_argument("--include-noise", action="store_true",
                       help="Process unmasked (noise) region")
    parser.add_argument("--noise-shrink", type=float, default=0.5,
                       help="Shrinkage factor for noise region")
    
    # Weighting parameters
    parser.add_argument("--weighting", type=str, default="uniform",
                       choices=["uniform", "performance", "cluster"],
                       help="Task weighting strategy")
    parser.add_argument("--performance-file", type=str, default=None,
                       help="Path to performance metrics JSON file")
    parser.add_argument("--temperature", type=float, default=5.0,
                       help="Temperature for performance-based weighting")
    parser.add_argument("--cluster-k", type=int, default=2,
                       help="Number of clusters for cluster-based weighting")
    
    # Storage and evaluation
    parser.add_argument("--store-artifacts", action="store_true",
                       help="Store compression artifacts")
    parser.add_argument("--eval-reconstruction", action="store_true", default=True,
                       help="Evaluate reconstruction error")
    parser.add_argument("--no-eval-reconstruction", action="store_false",
                       dest="eval_reconstruction",
                       help="Skip reconstruction evaluation")
    
    # Output paths
    parser.add_argument("--output-dir", type=str, default="./svd_hybrid_output",
                       help="Output directory for merged model")
    parser.add_argument("--artifact-dir", type=str, default="./artifacts",
                       help="Directory for artifact storage")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Load JSON config if specified
    import json
    config_overrides = {}
    
    if args.config:
        with open(args.config, 'r') as f:
            config_overrides = json.load(f)
        print(f"Loaded config from {args.config}")
    
    if args.quantize_config:
        with open(args.quantize_config, 'r') as f:
            quant_config = json.load(f)
        print(f"Loaded quantization config from {args.quantize_config}")
        # Merge quantization settings
        if 'quantization' in quant_config:
            config_overrides.update(quant_config['quantization'])
        if 'tasks' in quant_config and not args.tasks:
            config_overrides['tasks'] = quant_config['tasks']
        if 'checkpoints' in quant_config:
            config_overrides.update(quant_config['checkpoints'])
    
    if args.load_config:
        with open(args.load_config, 'r') as f:
            load_cfg = json.load(f)
        print(f"Loaded loading config from {args.load_config}")
        # Merge loading settings
        if 'loading' in load_cfg:
            config_overrides.update(load_cfg['loading'])
        if 'tasks' in load_cfg and not args.tasks:
            config_overrides['tasks'] = load_cfg['tasks']
        if 'checkpoints' in load_cfg:
            config_overrides.update(load_cfg['checkpoints'])
    
    # Apply config file values, then override with command-line args
    tasks = config_overrides.get('tasks', args.tasks)
    checkpoint_dir = config_overrides.get('checkpoint_dir', args.checkpoint_dir)
    base_model_path = config_overrides.get('base_model_path', args.base_model_path)
    
    if not tasks:
        raise ValueError("--tasks must be specified either via command-line or config file")
    if not checkpoint_dir:
        raise ValueError("--checkpoint-dir must be specified either via command-line or config file")
    if not base_model_path:
        raise ValueError("--base-model-path must be specified either via command-line or config file")
    
    # Create config from args
    config = SVDHybridConfig(
        tasks=tasks,
        model=args.model,
        checkpoint_dir=checkpoint_dir,
        base_model_path=base_model_path,
        mask_dir=args.mask_dir,
        svd_energy_threshold=args.energy_threshold,
        svd_max_rank=args.max_rank,
        svd_center=args.center,
        svd_fp16=args.fp16,
        svd_low_bits=args.low_bits,
        svd_rtvq_stages=args.rtvq_stages,
        svd_mask_strategy=args.mask_strategy,
        svd_include_noise=args.include_noise,
        svd_noise_shrink=args.noise_shrink,
        svd_weighting=args.weighting,
        performance_file=args.performance_file,
        svd_weighting_temperature=args.temperature,
        svd_cluster_k=args.cluster_k,
        svd_store_artifacts=args.store_artifacts,
        svd_eval_reconstruction=args.eval_reconstruction,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        device=args.device
    )
    
    # Run pipeline
    results = run_svd_hybrid_pipeline(config)
    
    return results


if __name__ == "__main__":
    main()
