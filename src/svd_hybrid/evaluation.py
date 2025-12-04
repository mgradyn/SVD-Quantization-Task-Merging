"""
Evaluation module for merged models.

This module provides tools to evaluate merged models on test datasets
and compare SVD-Hybrid results with baseline methods.

=== TUTORIAL: Model Evaluation ===

After merging models, you need to verify that the merged model actually works!
This module helps you:

1. Evaluate the merged model on each task's test set
2. Compare performance with baseline methods (Task Arithmetic, TIES, etc.)
3. Generate comprehensive reports showing what worked and what didn't

=== BASELINE METHODS ===

We compare SVD-Hybrid against these common merging approaches:

1. **Task Arithmetic**: Simple averaging of task vectors
   merged = base + avg(task_vectors)

2. **TIES Merging**: Trim, Elect, Sign - prunes small changes
   merged = base + ties_merge(task_vectors)

3. **Individual Models**: Using each fine-tuned model separately
   (upper bound on per-task performance)

=== EXAMPLE USAGE ===

    >>> from evaluation import evaluate_merged_model, compare_methods
    >>> 
    >>> # Evaluate a single merged model
    >>> results = evaluate_merged_model(
    ...     merged_state_dict=merged,
    ...     base_model_path="base.pt",
    ...     tasks=["Cars", "DTD", "EuroSAT"],
    ...     test_data_dir="./test_data"
    ... )
    >>> 
    >>> # Compare multiple merging methods
    >>> comparison = compare_methods(
    ...     base_model_path="base.pt",
    ...     task_checkpoint_paths={"Cars": "cars.pt", ...},
    ...     test_data_dir="./test_data"
    ... )
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path


def evaluate_merged_model(
    merged_state_dict: Dict[str, torch.Tensor],
    task_names: List[str],
    evaluation_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a merged model on multiple tasks.
    
    This is the main evaluation function that tests how well
    your merged model performs on each task.
    
    Args:
        merged_state_dict: State dict of the merged model
        task_names: List of task names to evaluate
        evaluation_fn: Custom evaluation function (task_name, state_dict) -> accuracy
                      If None, returns placeholder results
        verbose: Print tutorial-style progress messages
        
    Returns:
        Dictionary containing:
            - per_task_accuracy: Accuracy for each task
            - average_accuracy: Mean accuracy across tasks
            - validation_passed: Whether all checks passed
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📚 TUTORIAL: Evaluating Merged Model")
        print(f"{'='*70}")
        print(f"""
🎯 WHAT ARE WE DOING?
   We're testing how well the merged model performs on each task.
   
   A good merged model should:
   • Perform reasonably on ALL tasks (not just one)
   • Have accuracy close to the individual fine-tuned models
   • Show balanced performance (no task completely fails)
""")
    
    results = {
        "per_task_accuracy": {},
        "per_task_loss": {},
        "task_count": len(task_names),
        "validation_checks": {}
    }
    
    accuracies = []
    
    if verbose:
        print(f"📋 Tasks to evaluate: {task_names}")
        print(f"\n🔄 Running evaluation...")
    
    for i, task_name in enumerate(task_names):
        if verbose:
            print(f"\n   [{i+1}/{len(task_names)}] Evaluating on '{task_name}'...")
        
        if evaluation_fn is not None:
            try:
                accuracy = evaluation_fn(task_name, merged_state_dict)
                results["per_task_accuracy"][task_name] = accuracy
                accuracies.append(accuracy)
                
                if verbose:
                    bar_len = int(accuracy * 40)
                    bar = '█' * bar_len + '░' * (40 - bar_len)
                    print(f"      Accuracy: {accuracy*100:.2f}% [{bar}]")
                    
                    if accuracy >= 0.8:
                        print(f"      ✅ Good performance!")
                    elif accuracy >= 0.5:
                        print(f"      ⚠️ Moderate performance")
                    else:
                        print(f"      ❌ Low performance - may need investigation")
                        
            except Exception as e:
                if verbose:
                    print(f"      ❌ Evaluation failed: {e}")
                results["per_task_accuracy"][task_name] = None
        else:
            # No evaluation function provided - return placeholder
            if verbose:
                print(f"      ℹ️ No evaluation function provided - skipping actual evaluation")
                print(f"      💡 Provide evaluation_fn to get real accuracy metrics")
            results["per_task_accuracy"][task_name] = None
    
    # Compute statistics
    valid_accuracies = [a for a in accuracies if a is not None]
    
    if valid_accuracies:
        results["average_accuracy"] = np.mean(valid_accuracies)
        results["std_accuracy"] = np.std(valid_accuracies)
        results["min_accuracy"] = np.min(valid_accuracies)
        results["max_accuracy"] = np.max(valid_accuracies)
    else:
        results["average_accuracy"] = None
        results["std_accuracy"] = None
        results["min_accuracy"] = None
        results["max_accuracy"] = None
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'─'*70}")
        
        if results["average_accuracy"] is not None:
            print(f"""
   Overall Results:
   ├─ Average accuracy: {results['average_accuracy']*100:.2f}%
   ├─ Std deviation: {results['std_accuracy']*100:.2f}%
   ├─ Best task: {results['max_accuracy']*100:.2f}%
   └─ Worst task: {results['min_accuracy']*100:.2f}%
""")
            
            # Validation checks
            print(f"   🔬 VALIDATION CHECKS:")
            
            # Check 1: Average performance
            if results["average_accuracy"] >= 0.7:
                print(f"   ✅ CHECK 1 PASSED: Average accuracy ≥70%")
                results["validation_checks"]["average_ok"] = True
            else:
                print(f"   ⚠️ CHECK 1 WARNING: Average accuracy <70%")
                results["validation_checks"]["average_ok"] = False
            
            # Check 2: Balance (no task too far from average)
            if results["std_accuracy"] < 0.15:
                print(f"   ✅ CHECK 2 PASSED: Performance is balanced across tasks")
                results["validation_checks"]["balanced"] = True
            else:
                print(f"   ⚠️ CHECK 2 WARNING: Large variance between tasks")
                results["validation_checks"]["balanced"] = False
            
            # Check 3: No catastrophic failure
            if results["min_accuracy"] >= 0.3:
                print(f"   ✅ CHECK 3 PASSED: No task has catastrophic failure")
                results["validation_checks"]["no_failure"] = True
            else:
                print(f"   ❌ CHECK 3 FAILED: Some task(s) have very low accuracy")
                results["validation_checks"]["no_failure"] = False
            
            # Overall validation
            all_passed = all(results["validation_checks"].values())
            results["validation_passed"] = all_passed
            
            if all_passed:
                print(f"\n   🎉 ALL CHECKS PASSED - Merged model looks good!")
            else:
                print(f"\n   ⚠️ SOME CHECKS FAILED - Review results carefully")
        else:
            print(f"   ℹ️ No evaluation results (provide evaluation_fn for metrics)")
            results["validation_passed"] = None
    
    if verbose:
        print(f"\n{'='*70}\n")
    
    return results


def create_baseline_task_arithmetic(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    scaling_factor: float = 1.0,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Create a merged model using Task Arithmetic (simple averaging).
    
    This is the baseline method that simply averages all task vectors:
    merged = base + scaling_factor * mean(task_vectors)
    
    Args:
        base_state_dict: Base model state dict
        task_vectors: Dict mapping task_name -> param_name -> delta
        scaling_factor: Multiplier for the averaged task vector
        verbose: Print tutorial-style messages
        
    Returns:
        Merged model state dict
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📚 BASELINE: Task Arithmetic (Simple Averaging)")
        print(f"{'='*70}")
        print(f"""
🎯 HOW TASK ARITHMETIC WORKS:
   1. Compute task vectors: τ_i = fine_tuned_i - base
   2. Average them: τ_avg = (τ_1 + τ_2 + ... + τ_n) / n
   3. Apply to base: merged = base + scaling_factor × τ_avg
   
   Scaling factor: {scaling_factor}
   Number of tasks: {len(task_vectors)}
""")
    
    # Get all parameter names
    param_names = set()
    for tv in task_vectors.values():
        param_names.update(tv.keys())
    
    # Compute average task vector
    avg_task_vector = {}
    
    if verbose:
        print(f"   🔄 Computing average task vector...")
    
    for param_name in param_names:
        deltas = []
        for task_name, tv in task_vectors.items():
            if param_name in tv:
                deltas.append(tv[param_name])
        
        if deltas:
            avg_delta = torch.stack(deltas).mean(dim=0)
            avg_task_vector[param_name] = avg_delta * scaling_factor
    
    if verbose:
        avg_norm = sum(d.norm().item()**2 for d in avg_task_vector.values()) ** 0.5
        print(f"   ✅ Average task vector computed")
        print(f"      └─ Parameters: {len(avg_task_vector)}")
        print(f"      └─ Total norm: {avg_norm:.4f}")
    
    # Apply to base model
    if verbose:
        print(f"\n   🔄 Applying to base model...")
    
    merged = {}
    for param_name, base_param in base_state_dict.items():
        if param_name in avg_task_vector:
            merged[param_name] = base_param + avg_task_vector[param_name]
        else:
            merged[param_name] = base_param.clone()
    
    if verbose:
        print(f"   ✅ Task Arithmetic merge complete!")
        print(f"{'='*70}\n")
    
    return merged


def create_baseline_ties(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    density: float = 0.2,
    scaling_factor: float = 1.0,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Create a merged model using TIES (Trim, Elect Sign, Merge).
    
    TIES is a more sophisticated baseline that:
    1. Trims small magnitude changes (keeps top density%)
    2. Resolves sign conflicts by majority vote
    3. Averages the remaining values
    
    Args:
        base_state_dict: Base model state dict
        task_vectors: Dict mapping task_name -> param_name -> delta
        density: Fraction of parameters to keep (0.0-1.0)
        scaling_factor: Multiplier for the merged task vector
        verbose: Print tutorial-style messages
        
    Returns:
        Merged model state dict
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"📚 BASELINE: TIES Merging (Trim, Elect Sign, Merge)")
        print(f"{'='*70}")
        print(f"""
🎯 HOW TIES WORKS:
   TIES is smarter than simple averaging:
   
   1. TRIM: Remove small changes (keep top {density*100:.0f}% by magnitude)
      └─ Small changes are often noise, not useful signal
      
   2. ELECT SIGN: For each weight, vote on whether it should increase or decrease
      └─ Resolves conflicts when tasks disagree on direction
      
   3. MERGE: Average the surviving values with the elected sign
   
   Settings:
   ├─ Density: {density} (keep top {density*100:.0f}% of changes)
   ├─ Scaling factor: {scaling_factor}
   └─ Number of tasks: {len(task_vectors)}
""")
    
    # Get all parameter names
    param_names = set()
    for tv in task_vectors.values():
        param_names.update(tv.keys())
    
    merged_task_vector = {}
    
    if verbose:
        print(f"   🔄 Processing parameters with TIES...")
        total_kept = 0
        total_elements = 0
    
    for param_name in param_names:
        # Collect deltas for this parameter
        deltas = []
        for task_name, tv in task_vectors.items():
            if param_name in tv:
                deltas.append(tv[param_name])
        
        if not deltas:
            continue
        
        # Stack into tensor [num_tasks, *param_shape]
        stacked = torch.stack(deltas)
        
        # STEP 1: TRIM - Keep top k% by magnitude for each task
        trimmed = stacked.clone()
        for i in range(len(deltas)):
            flat = trimmed[i].abs().flatten()
            k = max(1, int(density * flat.numel()))
            threshold = torch.topk(flat, k).values[-1]
            mask = trimmed[i].abs() >= threshold
            trimmed[i] = trimmed[i] * mask.float()
        
        # STEP 2: ELECT SIGN - Majority vote on sign
        signs = torch.sign(trimmed)
        sign_sum = signs.sum(dim=0)
        elected_sign = torch.sign(sign_sum)
        # Handle ties (sign_sum == 0) by using positive
        elected_sign[elected_sign == 0] = 1
        
        # STEP 3: MERGE - Average magnitudes, apply elected sign
        magnitudes = trimmed.abs()
        # Only average non-zero values
        non_zero_count = (trimmed != 0).float().sum(dim=0)
        non_zero_count[non_zero_count == 0] = 1  # Avoid division by zero
        avg_magnitude = magnitudes.sum(dim=0) / non_zero_count
        
        # Apply elected sign and scaling
        merged_delta = elected_sign * avg_magnitude * scaling_factor
        merged_task_vector[param_name] = merged_delta
        
        if verbose:
            total_elements += merged_delta.numel()
            total_kept += (merged_delta != 0).sum().item()
    
    if verbose:
        kept_ratio = total_kept / max(total_elements, 1) * 100
        print(f"   ✅ TIES processing complete")
        print(f"      └─ Parameters: {len(merged_task_vector)}")
        print(f"      └─ Non-zero after TIES: {kept_ratio:.1f}%")
    
    # Apply to base model
    if verbose:
        print(f"\n   🔄 Applying to base model...")
    
    merged = {}
    for param_name, base_param in base_state_dict.items():
        if param_name in merged_task_vector:
            merged[param_name] = base_param + merged_task_vector[param_name]
        else:
            merged[param_name] = base_param.clone()
    
    if verbose:
        print(f"   ✅ TIES merge complete!")
        print(f"{'='*70}\n")
    
    return merged


def compare_methods(
    base_state_dict: Dict[str, torch.Tensor],
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    svd_hybrid_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    task_names: List[str] = None,
    evaluation_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict:
    """
    Compare SVD-Hybrid with baseline merging methods.
    
    This function creates merged models using different methods and
    compares their performance, helping you understand:
    - How much SVD-Hybrid improves over simple baselines
    - Which method works best for your specific tasks
    - Trade-offs between different approaches
    
    Args:
        base_state_dict: Base model state dict
        task_vectors: Dict mapping task_name -> param_name -> delta
        svd_hybrid_state_dict: Pre-computed SVD-Hybrid merged model (optional)
        task_names: List of task names for evaluation
        evaluation_fn: Function to evaluate models: (task_name, state_dict) -> accuracy
        verbose: Print tutorial-style progress
        
    Returns:
        Comparison results dictionary
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"📚 COMPARISON: SVD-Hybrid vs Baseline Methods")
        print(f"{'='*80}")
        print(f"""
🎯 WHY COMPARE METHODS?
   Different merging methods have different trade-offs:
   
   • Task Arithmetic: Simple and fast, but may have conflicts
   • TIES: Handles conflicts better, but discards information
   • SVD-Hybrid: Finds optimal basis, but more complex
   
   Comparing helps you choose the best method for your use case.
""")
    
    if task_names is None:
        task_names = list(task_vectors.keys())
    
    results = {
        "methods": {},
        "comparison": {}
    }
    
    # Method 1: Task Arithmetic
    if verbose:
        print(f"\n{'─'*80}")
        print(f"🔬 METHOD 1: Task Arithmetic")
        print(f"{'─'*80}")
    
    ta_merged = create_baseline_task_arithmetic(
        base_state_dict, task_vectors, scaling_factor=1.0, verbose=verbose
    )
    
    if evaluation_fn:
        ta_results = evaluate_merged_model(
            ta_merged, task_names, evaluation_fn, verbose=verbose
        )
        results["methods"]["task_arithmetic"] = ta_results
    else:
        results["methods"]["task_arithmetic"] = {"state_dict": ta_merged}
    
    # Method 2: TIES
    if verbose:
        print(f"\n{'─'*80}")
        print(f"🔬 METHOD 2: TIES Merging")
        print(f"{'─'*80}")
    
    ties_merged = create_baseline_ties(
        base_state_dict, task_vectors, density=0.2, scaling_factor=1.0, verbose=verbose
    )
    
    if evaluation_fn:
        ties_results = evaluate_merged_model(
            ties_merged, task_names, evaluation_fn, verbose=verbose
        )
        results["methods"]["ties"] = ties_results
    else:
        results["methods"]["ties"] = {"state_dict": ties_merged}
    
    # Method 3: SVD-Hybrid (if provided)
    if svd_hybrid_state_dict is not None:
        if verbose:
            print(f"\n{'─'*80}")
            print(f"🔬 METHOD 3: SVD-Hybrid")
            print(f"{'─'*80}")
        
        if evaluation_fn:
            svd_results = evaluate_merged_model(
                svd_hybrid_state_dict, task_names, evaluation_fn, verbose=verbose
            )
            results["methods"]["svd_hybrid"] = svd_results
        else:
            results["methods"]["svd_hybrid"] = {"state_dict": svd_hybrid_state_dict}
    
    # Generate comparison summary
    if verbose and evaluation_fn:
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n   Method Comparison (Average Accuracy):")
        print(f"   {'─'*50}")
        
        method_scores = {}
        for method_name, method_results in results["methods"].items():
            if "average_accuracy" in method_results and method_results["average_accuracy"] is not None:
                acc = method_results["average_accuracy"]
                method_scores[method_name] = acc
                bar_len = int(acc * 40)
                bar = '█' * bar_len + '░' * (40 - bar_len)
                print(f"   {method_name:20s}: {acc*100:6.2f}% [{bar}]")
        
        if method_scores:
            best_method = max(method_scores.keys(), key=lambda x: method_scores[x])
            worst_method = min(method_scores.keys(), key=lambda x: method_scores[x])
            
            print(f"\n   🏆 Best method: {best_method} ({method_scores[best_method]*100:.2f}%)")
            print(f"   📉 Worst method: {worst_method} ({method_scores[worst_method]*100:.2f}%)")
            
            if "svd_hybrid" in method_scores and "task_arithmetic" in method_scores:
                improvement = method_scores["svd_hybrid"] - method_scores["task_arithmetic"]
                print(f"\n   📈 SVD-Hybrid vs Task Arithmetic: {improvement*100:+.2f}%")
                
                if improvement > 0.02:
                    print(f"      ✅ SVD-Hybrid shows meaningful improvement!")
                elif improvement > -0.02:
                    print(f"      ≈ Methods perform similarly")
                else:
                    print(f"      ⚠️ Task Arithmetic performed better (unusual)")
            
            results["comparison"]["best_method"] = best_method
            results["comparison"]["scores"] = method_scores
    
    if verbose:
        print(f"""
   💡 INTERPRETATION GUIDE:
   • If SVD-Hybrid >> Task Arithmetic: Complex merging helps
   • If TIES >> Task Arithmetic: Sign conflicts were an issue
   • If all similar: Tasks may be naturally compatible
   • If all low: Consider different scaling factors or more tasks
""")
        print(f"{'='*80}\n")
    
    return results


def generate_evaluation_report(
    comparison_results: Dict,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Generate a human-readable evaluation report.
    
    Args:
        comparison_results: Results from compare_methods()
        output_path: Path to save the report (optional)
        verbose: Print the report
        
    Returns:
        Report as a string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SVD-HYBRID EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Method comparison
    if "methods" in comparison_results:
        lines.append("METHOD COMPARISON")
        lines.append("-" * 40)
        
        for method_name, results in comparison_results["methods"].items():
            lines.append(f"\n{method_name.upper()}:")
            if "average_accuracy" in results and results["average_accuracy"] is not None:
                lines.append(f"  Average Accuracy: {results['average_accuracy']*100:.2f}%")
                if "std_accuracy" in results:
                    lines.append(f"  Std Deviation: {results['std_accuracy']*100:.2f}%")
                if "per_task_accuracy" in results:
                    lines.append("  Per-task:")
                    for task, acc in results["per_task_accuracy"].items():
                        if acc is not None:
                            lines.append(f"    - {task}: {acc*100:.2f}%")
    
    # Summary
    if "comparison" in comparison_results and comparison_results["comparison"]:
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        if "best_method" in comparison_results["comparison"]:
            lines.append(f"Best Method: {comparison_results['comparison']['best_method']}")
    
    lines.append("")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    if verbose:
        print(report)
    
    return report
