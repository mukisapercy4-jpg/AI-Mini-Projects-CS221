"""
CS221 Assignment 1: Foundations Implementation
Author: Henry (Group AI Mini-Projects)
"""

import math
import re
from typing import List, Tuple, Dict, Any, Optional

def dot_product(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    """
    Computes the dot product of two sparse vectors represented as dictionaries.
    Optimized to iterate through the smaller dictionary for O(min(|d1|, |d2|)) efficiency.
    """
    if len(d1) > len(d2):
        d1, d2 = d2, d1
    return sum(val * d2.get(key, 0.0) for key, val in d1.items())

def increment_sparse_vector(d1: Dict[str, float], scale: float, d2: Dict[str, float]) -> None:
    """
    In-place update of sparse vector: d1 += scale * d2.
    """
    for key, val in d2.items():
        d1[key] = d1.get(key, 0.0) + scale * val

def compute_word_frequencies(sentence: str) -> Dict[str, int]:
    """
    Tokenizes input string into lowercase alphanumeric words and counts frequencies.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", sentence.lower())
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts

def find_longest_word(text: str) -> str:
    """
    Finds the longest token by character length in the input text.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text)
    if not tokens:
        return ""
    return max(tokens, key=len)

def compute_mle_bernoulli(trials: List[int]) -> float:
    """
    Computes the Maximum Likelihood Estimate (MLE) for Bernoulli parameter p.
    Formula: p_mle = (1 / n) * sum(x_i)
    """
    if not trials:
        return 0.0
    return sum(trials) / len(trials)

def compute_mle_gaussian(samples: List[float]) -> Tuple[float, float]:
    """
    Computes Maximum Likelihood Estimates (mu_mle, sigma_squared_mle) for Gaussian data.
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    mu = sum(samples) / n
    var = sum((x - mu) ** 2 for x in samples) / n
    return mu, var

def ridge_regression_gradient(
    w: Dict[str, float],
    points: List[Tuple[Dict[str, float], float]],
    reg_lambda: float
) -> Dict[str, float]:
    """
    Computes the analytical gradient of L2 regularized squared loss:
    grad = (1/n) * sum_i (w . x_i - y_i) * x_i + lambda * w
    """
    n = len(points)
    grad: Dict[str, float] = {}
    
    # Regularization gradient component: lambda * w
    for k, v in w.items():
        if v != 0.0:
            grad[k] = reg_lambda * v

    if n == 0:
        return grad

    scale = 1.0 / n
    for x_i, y_i in points:
        pred = dot_product(w, x_i)
        residual = pred - y_i
        increment_sparse_vector(grad, scale * residual, x_i)

    return grad

def gradient_descent_ridge(
    points: List[Tuple[Dict[str, float], float]],
    num_iters: int = 100,
    eta: float = 0.05,
    reg_lambda: float = 0.01
) -> Dict[str, float]:
    """
    Performs gradient descent to fit linear regression with L2 regularization.
    """
    w: Dict[str, float] = {}
    for _ in range(num_iters):
        grad = ridge_regression_gradient(w, points, reg_lambda)
        increment_sparse_vector(w, -eta, grad)
    return w
