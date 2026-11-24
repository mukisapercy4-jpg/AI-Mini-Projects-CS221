"""
Foundations Experiment & Benchmark Runner
Tests execution latency and convergence properties of sparse operations vs gradient descent.
"""

import time
from submission_foundations import (
    dot_product,
    increment_sparse_vector,
    gradient_descent_ridge
)
from foundations_engine import generate_synthetic_regression_data

def run_benchmarks():
    print("=" * 60)
    print("CS221 ASSIGNMENT 1: FOUNDATIONS BENCHMARKS")
    print("=" * 60)

    # 1. Sparse Vector Scaling Benchmark
    large_v1 = {f"feat_{i}": float(i) for i in range(10000)}
    small_v2 = {f"feat_{i * 10}": 1.5 for i in range(500)}

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = dot_product(large_v1, small_v2)
    dt_dot = (time.perf_counter() - t0) * 1000.0
    print(f"[BENCHMARK] 1,000 Sparse Dot Products (10,000 x 500 dims): {dt_dot:.2f} ms")

    # 2. Ridge Regression Optimization
    print("\n[EXPERIMENT] Fitting Ridge Regression Model across Iterations:")
    data, _ = generate_synthetic_regression_data(num_samples=250, vocab_size=30, seed=42)
    
    for iters in [20, 50, 100, 200]:
        t_start = time.perf_counter()
        w = gradient_descent_ridge(data, num_iters=iters, eta=0.08, reg_lambda=0.02)
        dur = (time.perf_counter() - t_start) * 1000.0
        
        # Calculate MSE
        mse = sum((dot_product(w, x) - y) ** 2 for x, y in data) / len(data)
        l2_norm = sum(v ** 2 for v in w.values())
        print(f"  Iterations: {iters:3d} | Train MSE: {mse:.4f} | ||w||^2: {l2_norm:.3f} | Elapsed: {dur:.2f} ms")

    print("=" * 60)
    print("Foundations experiment completed successfully.")

if __name__ == "__main__":
    run_benchmarks()
