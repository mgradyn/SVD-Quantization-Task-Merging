"""
Task clustering utilities for multi-basis merging.

=== TUTORIAL: Why Cluster Tasks? ===

When merging many task vectors, some tasks may be more similar to each other
than others. For example:

- Image classification tasks (Cars, DTD) might be similar
- Digit/character recognition (MNIST, SVHN) might be similar
- Scene understanding (SUN397, EuroSAT) might be similar

By grouping similar tasks into clusters, we can:

1. **Improve merging quality**: Similar tasks can share more structure
2. **Reduce interference**: Different task types don't interfere as much
3. **Better weighting**: Equal weight within groups, variable across groups

=== CLUSTERING METHODS ===

1. **K-Means**: Fast, partitions into k spherical clusters
   - Good for quick clustering
   - Requires specifying k upfront
   
2. **Hierarchical (Ward's method)**: Builds tree of clusters
   - Better for understanding task relationships
   - Can cut tree at different levels

=== HOW IT WORKS ===

1. Flatten each task's vector into a single feature vector
2. Normalize vectors (optional, improves clustering)
3. Apply clustering algorithm (k-means or hierarchical)
4. Return cluster assignments for each task

=== EXAMPLE ===

    >>> from clustering import cluster_tasks, get_cluster_members
    >>> 
    >>> # Assume task_vectors is already loaded
    >>> assignments = cluster_tasks(task_vectors, k=3, method="kmeans")
    >>> # {'Cars': 0, 'DTD': 0, 'EuroSAT': 1, 'SUN397': 1, 'MNIST': 2, 'SVHN': 2}
    >>> 
    >>> clusters = get_cluster_members(assignments)
    >>> # {0: ['Cars', 'DTD'], 1: ['EuroSAT', 'SUN397'], 2: ['MNIST', 'SVHN']}
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
    
    Each task vector is a dictionary of parameter tensors. This function
    concatenates all parameters into a single feature vector per task,
    creating a matrix suitable for clustering algorithms.
    
    === OUTPUT FORMAT ===
    
    Feature matrix: [N x D] where:
    - N = number of tasks
    - D = total flattened parameters (sum of all parameter sizes)
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        
    Returns:
        Tuple of:
            - Feature matrix as numpy array [N x D]
            - List of task names in the same order as matrix rows
            
    Example:
        >>> features, task_names = flatten_task_vectors(task_vectors)
        >>> features.shape
        (8, 1000000)  # 8 tasks, 1M total parameters
        >>> task_names
        ['Cars', 'DTD', 'EuroSAT', ...]
    """
    # Sort task names for consistent ordering
    task_names = sorted(task_vectors.keys())
    
    # Get all parameter names across all tasks
    all_params = set()
    for task_vector in task_vectors.values():
        all_params.update(task_vector.keys())
    param_names = sorted(all_params)
    
    # Flatten each task vector into a single row
    flattened_list = []
    for task_name in task_names:
        task_vector = task_vectors[task_name]
        
        # Concatenate all parameters for this task
        param_tensors = []
        for param_name in param_names:
            if param_name in task_vector:
                # Flatten this parameter tensor
                param_tensors.append(task_vector[param_name].flatten())
            else:
                # Task doesn't have this parameter, use zeros
                # Find the shape from another task that has it
                ref_task = next(t for t in task_vectors.values() if param_name in t)
                zeros = torch.zeros_like(ref_task[param_name]).flatten()
                param_tensors.append(zeros)
        
        # Concatenate all parameter tensors into one long vector
        flattened = torch.cat(param_tensors, dim=0)
        flattened_list.append(flattened.cpu().numpy())
    
    # Stack into matrix [N x D]
    feature_matrix = np.stack(flattened_list, axis=0)
    
    return feature_matrix, task_names


def compute_kmeans_clustering(
    features: np.ndarray,
    k: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute k-means clustering.
    
    Partitions the feature vectors into k clusters using the k-means algorithm.
    K-means minimizes the within-cluster sum of squares (variance).
    
    Args:
        features: Feature matrix [N x D] where N=samples, D=dimensions
        k: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Cluster assignments as numpy array [N], values in [0, k-1]
        
    Example:
        >>> features = np.random.randn(8, 1000)  # 8 tasks, 1000 features
        >>> labels = compute_kmeans_clustering(features, k=3)
        >>> labels
        array([0, 0, 1, 1, 2, 2, 0, 1])
    """
    if k <= 0 or k > features.shape[0]:
        raise ValueError(f"Invalid k={k} for {features.shape[0]} samples")
    
    # Create and fit k-means model
    # n_init=10 means 10 random initializations, take best
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
    
    Builds a tree (dendrogram) of clusters and cuts it to get k clusters.
    Ward's method minimizes within-cluster variance (similar to k-means objective).
    
    === LINKAGE METHODS ===
    
    - "ward": Minimize variance (recommended)
    - "complete": Maximum distance between clusters
    - "average": Average distance between clusters
    - "single": Minimum distance between clusters
    
    Args:
        features: Feature matrix [N x D]
        k: Number of clusters
        method: Linkage method for hierarchical clustering
        
    Returns:
        Cluster assignments as numpy array [N], values in [0, k-1]
    """
    if k <= 0 or k > features.shape[0]:
        raise ValueError(f"Invalid k={k} for {features.shape[0]} samples")
    
    # Compute linkage matrix (hierarchical structure)
    Z = linkage(features, method=method)
    
    # Cut dendrogram to get k clusters
    # fcluster returns 1-indexed labels, so subtract 1 for 0-indexed
    labels = fcluster(Z, k, criterion='maxclust') - 1
    
    return labels


def cluster_tasks(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    k: int,
    method: str = "kmeans"
) -> Dict[str, int]:
    """
    Cluster tasks based on their task vectors.
    
    Main entry point for task clustering. Flattens task vectors into feature
    vectors, applies clustering, and returns assignments.
    
    === PREPROCESSING ===
    
    Task vectors are L2-normalized before clustering, which helps when
    tasks have different magnitudes but similar directions.
    
    Args:
        task_vectors: Dictionary mapping task_name -> parameter_name -> delta tensor
        k: Number of clusters to create
        method: Clustering method - "kmeans" or "hierarchical"
        
    Returns:
        Dictionary mapping task_name -> cluster_id (0 to k-1)
        
    Example:
        >>> assignments = cluster_tasks(task_vectors, k=3, method="kmeans")
        >>> assignments
        {'Cars': 0, 'DTD': 0, 'EuroSAT': 1, 'SUN397': 1, 'MNIST': 2}
    """
    # Flatten task vectors into feature matrix
    features, task_names = flatten_task_vectors(task_vectors)
    
    # Normalize features (L2 normalization per task)
    # This makes clustering focus on direction rather than magnitude
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Apply clustering algorithm
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
    
    Inverts the cluster assignment dictionary to get members per cluster.
    Useful for understanding cluster composition.
    
    Args:
        cluster_assignments: Dictionary mapping task_name -> cluster_id
        
    Returns:
        Dictionary mapping cluster_id -> list of task names in that cluster
        
    Example:
        >>> assignments = {'Cars': 0, 'DTD': 0, 'EuroSAT': 1}
        >>> members = get_cluster_members(assignments)
        >>> members
        {0: ['Cars', 'DTD'], 1: ['EuroSAT']}
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
