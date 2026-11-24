"""
CS221 Assignment 2: Sentiment Classification Implementation
Author: Group AI Mini-Projects
"""

import math
import re
import collections
import random
from typing import List, Tuple, Dict, Any, Callable

def extract_word_features(x: str) -> Dict[str, float]:
    """
    Extracts unigram word count features from text string x.
    Converts to lowercase and tokenizes on alphanumeric boundaries.
    """
    features: Dict[str, float] = collections.defaultdict(float)
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", x.lower())
    for token in tokens:
        features[token] += 1.0
    return dict(features)

def dot_product(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    """Computes sparse dot product."""
    if len(d1) > len(d2):
        d1, d2 = d2, d1
    return sum(val * d2.get(key, 0.0) for key, val in d1.items())

def increment_sparse_vector(d1: Dict[str, float], scale: float, d2: Dict[str, float]) -> None:
    """In-place addition: d1 += scale * d2."""
    for key, val in d2.items():
        d1[key] = d1.get(key, 0.0) + scale * val

def predict_linear(w: Dict[str, float], phi: Dict[str, float]) -> int:
    """
    Predicts binary label (+1 or -1) using sign of linear score:
    y_hat = +1 if w . phi(x) >= 0 else -1
    """
    score = dot_product(w, phi)
    return 1 if score >= 0.0 else -1

def evaluate_classifier(
    w: Dict[str, float],
    dataset: List[Tuple[Dict[str, float], int]]
) -> float:
    """
    Computes classification error rate: (1/n) * sum(1[y_hat != y]).
    """
    if not dataset:
        return 0.0
    errors = sum(1 for phi, y in dataset if predict_linear(w, phi) != y)
    return errors / len(dataset)

def learn_predictor_sgd(
    train_data: List[Tuple[Dict[str, float], int]],
    dev_data: List[Tuple[Dict[str, float], int]],
    num_epochs: int = 15,
    eta: float = 0.05
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """
    Trains linear predictor using Stochastic Gradient Descent on Hinge Loss:
    Loss(x, y, w) = max(0, 1 - w . phi(x) * y)
    Update on margin violation (1 - margin > 0): w <- w + eta * y * phi(x)
    """
    w: Dict[str, float] = {}
    train_errors: List[float] = []
    dev_errors: List[float] = []

    for epoch in range(num_epochs):
        # Shuffle training examples per epoch
        shuffled = list(train_data)
        random.shuffle(shuffled)
        
        for phi, y in shuffled:
            margin = dot_product(w, phi) * y
            if margin < 1.0:
                # Subgradient update: w = w + eta * y * phi
                increment_sparse_vector(w, eta * y, phi)

        train_err = evaluate_classifier(w, train_data)
        dev_err = evaluate_classifier(w, dev_data)
        train_errors.append(train_err)
        dev_errors.append(dev_err)

    return w, train_errors, dev_errors

# =============================================================================
# K-MEANS CLUSTERING (UNSUPERVISED LEARNING)
# =============================================================================

def sparse_squared_distance(phi: Dict[str, float], mu: Dict[str, float]) -> float:
    """
    Computes squared Euclidean distance: ||phi - mu||^2 = ||phi||^2 + ||mu||^2 - 2 (phi . mu).
    """
    keys = set(phi.keys()).union(mu.keys())
    return sum((phi.get(k, 0.0) - mu.get(k, 0.0)) ** 2 for k in keys)

def kmeans_clustering(
    points: List[Dict[str, float]],
    k: int = 2,
    num_iters: int = 20,
    seed: int = 42
) -> Tuple[List[Dict[str, float]], List[int], List[float]]:
    """
    Implements K-Means Clustering on sparse feature representations.
    Returns: (centroids, cluster_assignments, distortion_history).
    """
    random.seed(seed)
    n = len(points)
    if n == 0 or k <= 0:
        return [], [], []

    # Initialize centroids randomly from points
    sampled_indices = random.sample(range(n), min(k, n))
    centroids = [{k: v for k, v in points[idx].items()} for idx in sampled_indices]
    
    assignments = [0] * n
    distortion_history: List[float] = []

    for _ in range(num_iters):
        # 1. Assignment Step
        total_distortion = 0.0
        for i, pt in enumerate(points):
            best_cluster = 0
            best_dist = float("inf")
            for c_idx, mu in enumerate(centroids):
                dist = sparse_squared_distance(pt, mu)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c_idx
            assignments[i] = best_cluster
            total_distortion += best_dist
        distortion_history.append(total_distortion)

        # 2. Update Centroid Step
        new_centroids: List[Dict[str, float]] = []
        for c_idx in range(k):
            members = [points[i] for i in range(n) if assignments[i] == c_idx]
            if not members:
                # Keep existing centroid if cluster is empty
                new_centroids.append(centroids[c_idx])
                continue
            
            centroid: Dict[str, float] = collections.defaultdict(float)
            scale = 1.0 / len(members)
            for m in members:
                increment_sparse_vector(centroid, scale, m)
            new_centroids.append(dict(centroid))

        centroids = new_centroids

    return centroids, assignments, distortion_history
