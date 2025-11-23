"""
Task clustering utilities for multi-basis merging.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster


def flatten_task_vectors(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Flatten all task vectors into a matrix for clustering.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        
    Returns:
        Tuple of (feature matrix [N x D], task names in order)
    """
    task_names = sorted(task_vectors.keys())
    
    # Get all parameter names
    all_params = set()
    for task_vector in task_vectors.values():
        all_params.update(task_vector.keys())
    param_names = sorted(all_params)
    
    # Flatten each task vector
    flattened_list = []
    for task_name in task_names:
        task_vector = task_vectors[task_name]
        
        # Concatenate all parameters for this task
        param_tensors = []
        for param_name in param_names:
            if param_name in task_vector:
                param_tensors.append(task_vector[param_name].flatten())
            else:
                # Task doesn't have this parameter, use zeros
                # Assume shape from another task
                ref_task = next(t for t in task_vectors.values() if param_name in t)
                zeros = torch.zeros_like(ref_task[param_name]).flatten()
                param_tensors.append(zeros)
        
        flattened = torch.cat(param_tensors, dim=0)
        flattened_list.append(flattened.cpu().numpy())
    
    feature_matrix = np.stack(flattened_list, axis=0)
    
    return feature_matrix, task_names


def compute_kmeans_clustering(
    features: np.ndarray,
    k: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute k-means clustering.
    
    Args:
        features: Feature matrix [N x D]
        k: Number of clusters
        random_state: Random seed
        
    Returns:
        Cluster assignments [N]
    """
    if k <= 0 or k > features.shape[0]:
        raise ValueError(f"Invalid k={k} for {features.shape[0]} samples")
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels


def compute_hierarchical_clustering(
    features: np.ndarray,
    k: int,
    method: str = "ward"
) -> np.ndarray:
    """
    Compute hierarchical clustering.
    
    Args:
        features: Feature matrix [N x D]
        k: Number of clusters
        method: Linkage method (ward, complete, average, single)
        
    Returns:
        Cluster assignments [N]
    """
    if k <= 0 or k > features.shape[0]:
        raise ValueError(f"Invalid k={k} for {features.shape[0]} samples")
    
    # Compute linkage
    Z = linkage(features, method=method)
    
    # Cut dendrogram to get k clusters
    labels = fcluster(Z, k, criterion='maxclust') - 1  # Make 0-indexed
    
    return labels


def cluster_tasks(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    k: int,
    method: str = "kmeans"
) -> Dict[str, int]:
    """
    Cluster tasks based on their task vectors.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        k: Number of clusters
        method: Clustering method ("kmeans" or "hierarchical")
        
    Returns:
        Dictionary mapping task_name -> cluster_id
    """
    # Flatten task vectors
    features, task_names = flatten_task_vectors(task_vectors)
    
    # Normalize features (optional but often helpful)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Cluster
    if method == "kmeans":
        labels = compute_kmeans_clustering(features, k)
    elif method == "hierarchical":
        labels = compute_hierarchical_clustering(features, k)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create assignment dictionary
    assignments = {name: int(label) for name, label in zip(task_names, labels)}
    
    return assignments


def get_cluster_members(
    cluster_assignments: Dict[str, int]
) -> Dict[int, List[str]]:
    """
    Get list of tasks in each cluster.
    
    Args:
        cluster_assignments: Dictionary mapping task_name -> cluster_id
        
    Returns:
        Dictionary mapping cluster_id -> list of task names
    """
    clusters = {}
    for task_name, cluster_id in cluster_assignments.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(task_name)
    
    return clusters


def compute_cluster_statistics(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    cluster_assignments: Dict[str, int]
) -> Dict[int, Dict]:
    """
    Compute statistics about each cluster.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        cluster_assignments: Dictionary mapping task_name -> cluster_id
        
    Returns:
        Dictionary mapping cluster_id -> statistics
    """
    clusters = get_cluster_members(cluster_assignments)
    features, task_names = flatten_task_vectors(task_vectors)
    
    task_to_idx = {name: idx for idx, name in enumerate(task_names)}
    
    statistics = {}
    
    for cluster_id, member_names in clusters.items():
        # Get feature vectors for cluster members
        member_indices = [task_to_idx[name] for name in member_names]
        cluster_features = features[member_indices]
        
        # Compute statistics
        centroid = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        statistics[cluster_id] = {
            "size": len(member_names),
            "members": member_names,
            "mean_distance_to_centroid": float(distances.mean()),
            "max_distance_to_centroid": float(distances.max()),
            "min_distance_to_centroid": float(distances.min())
        }
    
    return statistics


def merge_by_cluster(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    cluster_assignments: Dict[str, int],
    weights: Dict[str, float],
    device: str = "cpu"
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Compute weighted average within each cluster.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        cluster_assignments: Dictionary mapping task_name -> cluster_id
        weights: Dictionary mapping task_name -> weight
        device: Device for computation
        
    Returns:
        Dictionary mapping cluster_id -> parameter_name -> merged delta tensor
    """
    clusters = get_cluster_members(cluster_assignments)
    cluster_merged = {}
    
    for cluster_id, member_names in clusters.items():
        # Get task vectors for cluster members
        cluster_vectors = {name: task_vectors[name] for name in member_names}
        
        # Get weights for cluster members (normalize within cluster)
        cluster_weights = {name: weights.get(name, 1.0) for name in member_names}
        total_weight = sum(cluster_weights.values())
        cluster_weights = {k: v / total_weight for k, v in cluster_weights.items()}
        
        # Merge parameters
        param_names = set()
        for tv in cluster_vectors.values():
            param_names.update(tv.keys())
        
        merged_params = {}
        for param_name in param_names:
            # Get tensors for this parameter from cluster members
            param_tensors = {}
            for task_name in member_names:
                if param_name in cluster_vectors[task_name]:
                    param_tensors[task_name] = cluster_vectors[task_name][param_name]
            
            if param_tensors:
                # Weighted average
                from .weighting import apply_weights_to_tensors
                merged_params[param_name] = apply_weights_to_tensors(
                    param_tensors, cluster_weights, device
                )
        
        cluster_merged[cluster_id] = merged_params
    
    return cluster_merged


def merge_cluster_results(
    cluster_merged: Dict[int, Dict[str, torch.Tensor]],
    cluster_performance: Dict[int, float],
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Merge results from multiple clusters.
    
    Args:
        cluster_merged: Dictionary mapping cluster_id -> parameter_name -> tensor
        cluster_performance: Dictionary mapping cluster_id -> performance metric
        device: Device for computation
        
    Returns:
        Final merged task vector
    """
    if not cluster_merged:
        return {}
    
    # Get all parameter names
    param_names = set()
    for params in cluster_merged.values():
        param_names.update(params.keys())
    
    # Compute cluster weights from performance
    cluster_ids = list(cluster_merged.keys())
    if cluster_performance:
        performances = torch.tensor([cluster_performance.get(cid, 1.0) for cid in cluster_ids])
        cluster_weights = torch.softmax(performances, dim=0)
    else:
        # Uniform weights
        cluster_weights = torch.ones(len(cluster_ids)) / len(cluster_ids)
    
    cluster_weight_dict = {cid: w.item() for cid, w in zip(cluster_ids, cluster_weights)}
    
    # Merge each parameter
    final_merged = {}
    for param_name in param_names:
        # Get tensors from each cluster
        param_tensors = {}
        for cluster_id in cluster_ids:
            if param_name in cluster_merged[cluster_id]:
                param_tensors[cluster_id] = cluster_merged[cluster_id][param_name]
        
        if param_tensors:
            # Weighted average across clusters
            from .weighting import apply_weights_to_tensors
            final_merged[param_name] = apply_weights_to_tensors(
                param_tensors, cluster_weight_dict, device
            )
    
    return final_merged
