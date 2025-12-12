"""
Synthetic data generation for benchmarks.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Literal


class DataGenerator:
    """
    Generate synthetic vector data for benchmarking.
    """
    
    @staticmethod
    def generate_vectors(
        count: int,
        dimensions: int,
        distribution: Literal['uniform', 'normal', 'clustered'] = 'normal',
        seed: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate synthetic vectors.
        
        Args:
            count: Number of vectors to generate
            dimensions: Vector dimensionality
            distribution: Distribution type ('uniform', 'normal', 'clustered')
            seed: Random seed for reproducibility
            normalize: Whether to L2-normalize vectors
        
        Returns:
            Array of shape (count, dimensions)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if distribution == 'uniform':
            vectors = np.random.uniform(-1.0, 1.0, size=(count, dimensions))
        elif distribution == 'normal':
            vectors = np.random.normal(0.0, 1.0, size=(count, dimensions))
        elif distribution == 'clustered':
            # Generate vectors in clusters
            num_clusters = max(1, count // 100)
            vectors = np.zeros((count, dimensions))
            cluster_centers = np.random.normal(0.0, 2.0, size=(num_clusters, dimensions))
            
            for i in range(count):
                cluster_idx = i % num_clusters
                # Add noise around cluster center
                vectors[i] = cluster_centers[cluster_idx] + np.random.normal(0.0, 0.3, size=dimensions)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        if normalize:
            # L2 normalize
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            vectors = vectors / norms
        
        return vectors
    
    @staticmethod
    def generate_query_vectors(
        count: int,
        dimensions: int,
        method: Literal['random', 'from_dataset', 'clustered'] = 'random',
        dataset_vectors: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate query vectors for testing.
        
        Args:
            count: Number of query vectors
            dimensions: Vector dimensionality
            method: Generation method
                - 'random': Generate random vectors
                - 'from_dataset': Sample from dataset (requires dataset_vectors)
                - 'clustered': Generate clustered queries
            dataset_vectors: Optional dataset to sample from
            seed: Random seed
            normalize: Whether to normalize vectors
        
        Returns:
            Array of shape (count, dimensions)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if method == 'random':
            queries = DataGenerator.generate_vectors(
                count, dimensions, 'normal', seed, normalize
            )
        elif method == 'from_dataset':
            if dataset_vectors is None:
                raise ValueError("dataset_vectors required for 'from_dataset' method")
            # Sample random vectors from dataset
            indices = np.random.choice(len(dataset_vectors), size=count, replace=True)
            queries = dataset_vectors[indices].copy()
            if normalize:
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                queries = queries / norms
        elif method == 'clustered':
            queries = DataGenerator.generate_vectors(
                count, dimensions, 'clustered', seed, normalize
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return queries
    
    @staticmethod
    def vectors_to_sql_format(vectors: np.ndarray) -> List[str]:
        """
        Convert numpy array of vectors to PostgreSQL vector literal format.
        
        Args:
            vectors: Array of shape (n, d)
        
        Returns:
            List of vector strings in format '[v1,v2,...,vd]'
        """
        return [
            '[' + ','.join(str(float(x)) for x in vec) + ']'
            for vec in vectors
        ]
    
    @staticmethod
    def compute_ground_truth(
        query_vector: np.ndarray,
        dataset_vectors: np.ndarray,
        k: int,
        metric: Literal['l2', 'cosine', 'inner_product'] = 'l2'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ground truth nearest neighbors.
        
        Args:
            query_vector: Query vector of shape (d,)
            dataset_vectors: Dataset vectors of shape (n, d)
            k: Number of neighbors to find
            metric: Distance metric
        
        Returns:
            Tuple of (indices, distances) for top-k neighbors
        """
        if metric == 'l2':
            distances = np.linalg.norm(dataset_vectors - query_vector, axis=1)
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            dot_products = np.dot(dataset_vectors, query_vector)
            norms = np.linalg.norm(dataset_vectors, axis=1) * np.linalg.norm(query_vector)
            cosine_similarities = dot_products / (norms + 1e-8)
            distances = 1.0 - cosine_similarities
        elif metric == 'inner_product':
            # Inner product (negative for ordering)
            distances = -np.dot(dataset_vectors, query_vector)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k]
        top_k_distances = distances[top_k_indices]
        
        return top_k_indices, top_k_distances
    
    @staticmethod
    def compute_recall(
        predicted_indices: np.ndarray,
        ground_truth_indices: np.ndarray
    ) -> float:
        """
        Compute recall@k.
        
        Args:
            predicted_indices: Indices returned by search
            ground_truth_indices: Ground truth indices
        
        Returns:
            Recall score (0.0 to 1.0)
        """
        if len(ground_truth_indices) == 0:
            return 1.0 if len(predicted_indices) == 0 else 0.0
        
        intersection = np.intersect1d(predicted_indices, ground_truth_indices)
        return len(intersection) / len(ground_truth_indices)

