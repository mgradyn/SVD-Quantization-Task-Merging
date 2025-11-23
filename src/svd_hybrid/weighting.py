"""
Task weighting utilities for performance-based and cluster-based merging.
"""
import torch
import json
from typing import Dict, List, Optional
import numpy as np


def load_performance_metrics(
    performance_file: str,
    task_names: List[str]
) -> Dict[str, float]:
    """
    Load per-task performance metrics from JSON file.
    
    Args:
        performance_file: Path to JSON file with performance metrics
        task_names: List of task names to load metrics for
        
    Returns:
        Dictionary mapping task_name -> accuracy/performance
    """
    with open(performance_file, 'r') as f:
        data = json.load(f)
    
    metrics = {}
    for task_name in task_names:
        # Try exact match first
        if task_name in data:
            metrics[task_name] = float(data[task_name])
        else:
            # Try fuzzy matching (case insensitive, underscore/dash agnostic)
            normalized_task = task_name.lower().replace('_', '').replace('-', '')
            found = False
            
            for key, value in data.items():
                normalized_key = key.lower().replace('_', '').replace('-', '')
                if normalized_key == normalized_task:
                    metrics[task_name] = float(value)
                    found = True
                    break
            
            if not found:
                print(f"Warning: No performance metric found for task {task_name}, using default 1.0")
                metrics[task_name] = 1.0
    
    return metrics


def compute_uniform_weights(task_names: List[str]) -> Dict[str, float]:
    """
    Compute uniform weights (all tasks equal).
    
    Args:
        task_names: List of task names
        
    Returns:
        Dictionary mapping task_name -> weight
    """
    weight = 1.0 / len(task_names)
    return {name: weight for name in task_names}


def compute_performance_weights(
    performance_metrics: Dict[str, float],
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Compute performance-based weights using softmax.
    
    Args:
        performance_metrics: Dictionary mapping task_name -> performance
        temperature: Temperature for softmax (higher = more uniform)
        
    Returns:
        Dictionary mapping task_name -> weight
    """
    if not performance_metrics:
        return {}
    
    task_names = list(performance_metrics.keys())
    performances = torch.tensor([performance_metrics[name] for name in task_names])
    
    # Apply temperature
    scaled = performances / temperature
    
    # Softmax
    weights = torch.softmax(scaled, dim=0)
    
    return {name: weight.item() for name, weight in zip(task_names, weights)}


def compute_cluster_weights(
    task_names: List[str],
    cluster_assignments: Dict[str, int],
    cluster_performance: Optional[Dict[int, float]] = None
) -> Dict[str, float]:
    """
    Compute weights based on cluster assignments.
    
    Within each cluster, tasks have equal weight.
    Optionally, clusters can be weighted by their mean performance.
    
    Args:
        task_names: List of task names
        cluster_assignments: Dictionary mapping task_name -> cluster_id
        cluster_performance: Optional dictionary mapping cluster_id -> performance
        
    Returns:
        Dictionary mapping task_name -> weight
    """
    # Count tasks per cluster
    cluster_counts = {}
    for task_name in task_names:
        cluster_id = cluster_assignments.get(task_name, 0)
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    # Compute cluster weights
    num_clusters = len(cluster_counts)
    
    if cluster_performance is not None:
        # Weight clusters by performance
        cluster_ids = list(cluster_counts.keys())
        performances = torch.tensor([cluster_performance.get(cid, 1.0) for cid in cluster_ids])
        cluster_weight_tensor = torch.softmax(performances, dim=0)
        cluster_weights = {cid: w.item() for cid, w in zip(cluster_ids, cluster_weight_tensor)}
    else:
        # Uniform cluster weights
        cluster_weights = {cid: 1.0 / num_clusters for cid in cluster_counts.keys()}
    
    # Distribute cluster weight equally among tasks in cluster
    task_weights = {}
    for task_name in task_names:
        cluster_id = cluster_assignments.get(task_name, 0)
        cluster_weight = cluster_weights[cluster_id]
        task_count = cluster_counts[cluster_id]
        task_weights[task_name] = cluster_weight / task_count
    
    # Normalize to sum to 1
    total = sum(task_weights.values())
    if total > 0:
        task_weights = {k: v / total for k, v in task_weights.items()}
    
    return task_weights


def compute_weights(
    task_names: List[str],
    weighting_strategy: str = "uniform",
    performance_file: Optional[str] = None,
    temperature: float = 1.0,
    cluster_assignments: Optional[Dict[str, int]] = None,
    cluster_performance: Optional[Dict[int, float]] = None
) -> Dict[str, float]:
    """
    Compute task weights based on specified strategy.
    
    Args:
        task_names: List of task names
        weighting_strategy: "uniform", "performance", or "cluster"
        performance_file: Path to performance metrics JSON (for "performance")
        temperature: Temperature for softmax (for "performance")
        cluster_assignments: Cluster assignments (for "cluster")
        cluster_performance: Cluster performance metrics (for "cluster")
        
    Returns:
        Dictionary mapping task_name -> weight
    """
    if weighting_strategy == "uniform":
        return compute_uniform_weights(task_names)
    
    elif weighting_strategy == "performance":
        if performance_file is None:
            print("Warning: No performance file provided, using uniform weights")
            return compute_uniform_weights(task_names)
        
        performance_metrics = load_performance_metrics(performance_file, task_names)
        return compute_performance_weights(performance_metrics, temperature)
    
    elif weighting_strategy == "cluster":
        if cluster_assignments is None:
            print("Warning: No cluster assignments provided, using uniform weights")
            return compute_uniform_weights(task_names)
        
        return compute_cluster_weights(task_names, cluster_assignments, cluster_performance)
    
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")


def apply_weights_to_tensors(
    tensors: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute weighted average of tensors.
    
    Args:
        tensors: Dictionary mapping task_name -> tensor
        weights: Dictionary mapping task_name -> weight
        device: Device for computation
        
    Returns:
        Weighted average tensor
    """
    if not tensors:
        raise ValueError("Empty tensor dictionary")
    
    # Ensure consistent keys
    task_names = sorted(tensors.keys())
    
    # Stack tensors
    tensor_list = [tensors[name].to(device).float() for name in task_names]
    stacked = torch.stack(tensor_list, dim=0)  # [N x ...]
    
    # Get weights in same order
    weight_list = [weights.get(name, 1.0 / len(task_names)) for name in task_names]
    weight_tensor = torch.tensor(weight_list, device=device, dtype=torch.float32)
    
    # Normalize weights
    weight_tensor = weight_tensor / weight_tensor.sum()
    
    # Reshape weights for broadcasting
    weight_shape = [len(task_names)] + [1] * (stacked.ndim - 1)
    weight_tensor = weight_tensor.view(weight_shape)
    
    # Compute weighted average
    weighted_avg = (stacked * weight_tensor).sum(dim=0)
    
    return weighted_avg


def get_weight_statistics(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute statistics about weight distribution.
    
    Args:
        weights: Dictionary mapping task_name -> weight
        
    Returns:
        Dictionary of statistics
    """
    if not weights:
        return {}
    
    values = list(weights.values())
    
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": np.std(values),
        "entropy": -sum(w * np.log(w + 1e-10) for w in values)
    }
