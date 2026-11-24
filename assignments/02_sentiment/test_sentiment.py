"""
Unit Test Suite for Assignment 2: Sentiment Classification
Verifies feature extraction, SGD on Hinge Loss, and K-Means clustering.
"""

from submission_sentiment import (
    extract_word_features,
    predict_linear,
    evaluate_classifier,
    learn_predictor_sgd,
    kmeans_clustering,
    sparse_squared_distance
)
from sentiment_engine import get_movie_review_dataset

def run_tests():
    print("Testing Feature Extraction...")
    text = "The cinematography was great and great!"
    feats = extract_word_features(text)
    assert feats["great"] == 2.0
    assert feats["cinematography"] == 1.0
    print("  [PASS] extract_word_features verified.")

    print("Testing Linear Predictor...")
    w = {"good": 2.0, "bad": -3.0}
    assert predict_linear(w, {"good": 1.0}) == 1
    assert predict_linear(w, {"bad": 1.0}) == -1
    assert predict_linear(w, {"good": 1.0, "bad": 1.0}) == -1
    print("  [PASS] predict_linear verified.")

    print("Testing Sentiment SGD Classifier...")
    raw_train, raw_dev = get_movie_review_dataset(num_train=40, num_dev=15, seed=10)
    train_data = [(extract_word_features(text), y) for text, y in raw_train]
    dev_data = [(extract_word_features(text), y) for text, y in raw_dev]

    w_opt, train_errs, dev_errs = learn_predictor_sgd(
        train_data, dev_data, num_epochs=20, eta=0.08
    )
    final_train_err = train_errs[-1]
    final_dev_err = dev_errs[-1]
    print(f"  [PASS] SGD trained. Final Train Error: {final_train_err*100:.1f}%, Dev Error: {final_dev_err*100:.1f}%")
    assert final_train_err < 0.25, f"Train error too high: {final_train_err}"

    print("Testing K-Means Clustering...")
    pts = [
        {"x": 1.0, "y": 1.0},
        {"x": 1.2, "y": 0.8},
        {"x": 10.0, "y": 10.0},
        {"x": 10.5, "y": 9.8}
    ]
    centroids, assignments, distortions = kmeans_clustering(pts, k=2, num_iters=10, seed=1)
    assert assignments[0] == assignments[1], "First two points should cluster together"
    assert assignments[2] == assignments[3], "Last two points should cluster together"
    assert assignments[0] != assignments[2], "Distinct clusters expected"
    # Check monotonicity of distortion
    for i in range(len(distortions) - 1):
        assert distortions[i+1] <= distortions[i] + 1e-9, "Distortion must non-increase"
    print("  [PASS] K-Means clustering verified.")

    print("\nALL BASIC SENTIMENT TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
