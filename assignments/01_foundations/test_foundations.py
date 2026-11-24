"""
Unit Test Suite for Assignment 1: Foundations
Verifies sparse vector operations, text helpers, MLE derivations, and gradient descent.
"""

import math
from submission_foundations import (
    dot_product,
    increment_sparse_vector,
    compute_word_frequencies,
    find_longest_word,
    compute_mle_bernoulli,
    compute_mle_gaussian,
    ridge_regression_gradient,
    gradient_descent_ridge
)
from foundations_engine import (
    generate_synthetic_regression_data,
    sample_bernoulli_trials,
    sample_gaussian_data
)

def run_tests():
    print("Testing Sparse Vector Dot Product...")
    v1 = {"apple": 2.0, "banana": 3.0, "cherry": 1.0}
    v2 = {"banana": 4.0, "cherry": -1.0, "dates": 5.0}
    # 2*0 + 3*4 + 1*(-1) + 0*5 = 11.0
    res = dot_product(v1, v2)
    assert abs(res - 11.0) < 1e-6, f"Expected 11.0, got {res}"
    print("  [PASS] dot_product verified.")

    print("Testing In-Place Sparse Vector Increment...")
    target = {"a": 1.0, "b": 2.0}
    source = {"b": 3.0, "c": 4.0}
    increment_sparse_vector(target, 2.0, source)
    assert abs(target["a"] - 1.0) < 1e-6
    assert abs(target["b"] - 8.0) < 1e-6
    assert abs(target["c"] - 8.0) < 1e-6
    print("  [PASS] increment_sparse_vector verified.")

    print("Testing Word Frequency and Longest Word...")
    text = "Artificial intelligence and machine learning empower intelligent agents."
    freqs = compute_word_frequencies(text)
    assert freqs["intelligence"] == 1
    assert freqs["and"] == 1
    longest = find_longest_word(text)
    assert longest.lower() == "intelligence", f"Expected intelligence, got {longest}"
    print("  [PASS] Word processing verified.")

    print("Testing Bernoulli and Gaussian MLE...")
    trials = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    p_hat = compute_mle_bernoulli(trials)
    assert abs(p_hat - 0.7) < 1e-6
    
    samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    mu_hat, var_hat = compute_mle_gaussian(samples)
    assert abs(mu_hat - 5.0) < 1e-6
    assert abs(var_hat - 4.0) < 1e-6
    print("  [PASS] MLE derivations verified.")

    print("Testing Ridge Regression Gradient Descent...")
    data, true_weights = generate_synthetic_regression_data(num_samples=150, seed=123)
    fitted_w = gradient_descent_ridge(data, num_iters=150, eta=0.1, reg_lambda=0.01)
    
    # Check that predictions reasonably correlate with targets
    total_loss = 0.0
    for x, y in data:
        pred = dot_product(fitted_w, x)
        total_loss += (pred - y) ** 2
    mse = total_loss / len(data)
    print(f"  [PASS] Model trained. Train MSE: {mse:.4f}")
    assert mse < 1.0, f"MSE too high: {mse}"

    print("\nALL BASIC FOUNDATIONS TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
