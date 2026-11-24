"""
Foundations Engine & Dataset Helpers
Provides synthetic dataset generation and baseline mathematical models for CS221 Assignment 1.
"""

import math
import random
from typing import List, Tuple, Dict, Any

def generate_synthetic_regression_data(
    num_samples: int = 100,
    vocab_size: int = 20,
    density: float = 0.25,
    noise_std: float = 0.05,
    seed: int = 42
) -> Tuple[List[Tuple[Dict[str, float], float]], Dict[str, float]]:
    """
    Generates sparse linear regression dataset: (x_i, y_i) where y_i = w^* . x_i + noise.
    """
    random.seed(seed)
    vocab = [f"feat_{i}" for i in range(vocab_size)]
    true_weights = {feat: round(random.uniform(-2.0, 2.0), 3) for feat in vocab if random.random() < 0.5}

    data = []
    for _ in range(num_samples):
        x = {}
        for feat in vocab:
            if random.random() < density:
                x[feat] = round(random.uniform(0.1, 1.0), 3)
        # Compute true dot product
        y = sum(x.get(k, 0.0) * val for k, val in true_weights.items()) + random.gauss(0, noise_std)
        data.append((x, round(y, 4)))

    return data, true_weights

def sample_bernoulli_trials(p_true: float = 0.65, n_trials: int = 200, seed: int = 42) -> List[int]:
    """Generates synthetic Bernoulli trial observations."""
    random.seed(seed)
    return [1 if random.random() < p_true else 0 for _ in range(n_trials)]

def sample_gaussian_data(mu_true: float = 5.0, sigma_true: float = 1.5, n_samples: int = 300, seed: int = 42) -> List[float]:
    """Generates synthetic 1D Gaussian samples."""
    random.seed(seed)
    return [round(random.gauss(mu_true, sigma_true), 4) for _ in range(n_samples)]
