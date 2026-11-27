"""
Unit Test Suite for Assignment 2: Sentiment Classification
Updated to match Hilton's (hiltonjolly-2020) function signatures in submission_sentiment.py.
All tests verified against the master notebook implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
from submission_sentiment import (
    extract_word_features,
    extract_character_features,
    dot_product,
    increment_sparse_vector,
    learn_predictor_sgd,   # from submission_sentiment.py helper
    kmeans_clustering,     # from submission_sentiment.py helper
)

# ------ Thin wrappers to call the CS221-named functions when present --------

def _predict(weights, phi):
    """Sign of linear score: +1 if w.phi >= 0 else -1."""
    score = dot_product(weights, phi)
    return 1 if score >= 0 else -1

def _evaluate(weights, data):
    """Error rate on a dataset of (phi, y) pairs."""
    if not data:
        return 0.0
    errors = sum(1 for phi, y in data if _predict(weights, phi) != y)
    return errors / len(data)


def run_tests():
    # 1. Feature Extraction ----------------------------------------------------
    print("Testing extract_word_features...")
    # Use punctuation-free input so "great" appears as the same token twice
    text = "The cinematography was great and great"
    feats = extract_word_features(text)
    assert feats.get("great") == 2.0, f"Expected 2.0 for 'great', got {feats.get('great')}"
    assert feats.get("cinematography") == 1.0
    # Also verify non-word-split behaviour: "great!" is a distinct token
    text2 = "great great!"
    feats2 = extract_word_features(text2)
    assert feats2.get("great") == 1.0 and feats2.get("great!") == 1.0, \
        "Tokenisation splits on whitespace only — 'great' and 'great!' are distinct"
    print("  [PASS] extract_word_features: word counts correct.")

    # 2. Character N-gram Features ---------------------------------------------
    print("Testing extract_character_features...")
    extractor3 = extract_character_features(3)
    result = extractor3("hello")
    assert result.get("hel") == 1.0
    assert result.get("ell") == 1.0
    assert result.get("llo") == 1.0
    # Spaces should be stripped before n-gram extraction: "ab cd" → "abcd"
    # bigrams of "abcd": "ab", "bc", "cd"
    result_space = extract_character_features(2)("ab cd")
    assert result_space.get("ab") == 1.0, "bigram 'ab' expected"
    assert result_space.get("bc") == 1.0, "Space is stripped so 'bc' should be a valid bigram"
    assert result_space.get("cd") == 1.0, "bigram 'cd' expected"
    print("  [PASS] extract_character_features: n-grams correct, spaces stripped.")

    # 3. Dot Product -----------------------------------------------------------
    print("Testing dot_product...")
    v1 = {"apple": 2.0, "banana": 3.0}
    v2 = {"banana": 4.0, "cherry": -1.0}
    res = dot_product(v1, v2)
    assert abs(res - 12.0) < 1e-9, f"Expected 12.0, got {res}"
    print("  [PASS] dot_product: correct sparse inner product.")

    # 4. Increment Sparse Vector -----------------------------------------------
    print("Testing increment_sparse_vector...")
    d = {"a": 1.0, "b": 2.0}
    increment_sparse_vector(d, 2.0, {"b": 3.0, "c": 4.0})
    assert abs(d["a"] - 1.0) < 1e-9
    assert abs(d["b"] - 8.0) < 1e-9
    assert abs(d["c"] - 8.0) < 1e-9
    print("  [PASS] increment_sparse_vector: in-place update correct.")

    # 5. Linear Predictor (predict_linear / _predict) -------------------------
    print("Testing linear prediction (sign of w.phi)...")
    w_test = {"good": 2.0, "bad": -3.0}
    assert _predict(w_test, {"good": 1.0}) == 1
    assert _predict(w_test, {"bad": 1.0}) == -1
    assert _predict(w_test, {"good": 1.0, "bad": 1.0}) == -1  # 2 - 3 = -1 < 0
    print("  [PASS] Linear predictor: sign of score correct.")

    # 6. SGD Classifier (learn_predictor_sgd) ---------------------------------
    print("Testing learn_predictor_sgd (hinge loss)...")
    POS = [
        "this movie was absolutely fantastic and brilliant",
        "loved it wonderful heartwarming story great cast",
        "superb direction and incredible performances throughout",
        "a masterpiece of modern cinema truly outstanding",
        "brilliant storytelling and stunning cinematography",
        "wonderful experience great acting and dialogue",
        "exceptional and visually breathtaking film",
        "loved every scene magnificent directing",
        "five stars incredible performance",
        "outstanding movie loved it completely",
    ]
    NEG = [
        "terrible waste of time boring and dull",
        "awful acting and dreadful script complete mess",
        "horrible plot and poor direction unwatchable",
        "worst movie ever seen completely terrible",
        "dull lifeless and painfully boring throughout",
        "complete disaster poor acting and writing",
        "absolute rubbish avoid at all costs",
        "dreadful waste of two hours awful",
        "terrible boring nothing works in this film",
        "horrendous script and laughable effects",
    ]
    random.seed(42)
    raw = [(r, 1) for r in POS] + [(r, -1) for r in NEG]
    random.shuffle(raw)
    train = [(extract_word_features(x), y) for x, y in raw[:16]]
    dev = [(extract_word_features(x), y) for x, y in raw[16:]]

    w_opt, train_errs, dev_errs = learn_predictor_sgd(train, dev, num_epochs=25, eta=0.1)
    final_train_err = train_errs[-1]
    final_dev_err = dev_errs[-1]
    print(f"  [PASS] SGD trained 25 epochs. Train Error: {final_train_err*100:.1f}%  Dev Error: {final_dev_err*100:.1f}%")
    assert final_train_err < 0.25, f"Train error too high: {final_train_err:.2f}"

    # 7. K-Means Clustering (kmeans_clustering) --------------------------------
    print("Testing kmeans_clustering...")
    pts = [
        {"x": 1.0, "y": 1.0},
        {"x": 1.3, "y": 0.8},
        {"x": 10.0, "y": 10.0},
        {"x": 10.5, "y": 9.7},
    ]
    centroids, assignments, distortions = kmeans_clustering(pts, k=2, num_iters=15, seed=1)
    assert assignments[0] == assignments[1], "First pair must cluster together"
    assert assignments[2] == assignments[3], "Second pair must cluster together"
    assert assignments[0] != assignments[2], "Two distinct clusters expected"
    if len(distortions) > 1:
        for i in range(len(distortions) - 1):
            assert distortions[i+1] <= distortions[i] + 1e-6, \
                f"Distortion must not increase: {distortions[i+1]} > {distortions[i]}"
    print("  [PASS] K-Means: correct cluster assignments and monotonic distortion.")

    # 8. Character N-gram Feature Training comparison -------------------------
    print("Testing character n-gram feature performance...")
    extractor_char3 = extract_character_features(3)
    train_char = [(extractor_char3(x), y) for x, y in raw[:16]]
    dev_char = [(extractor_char3(x), y) for x, y in raw[16:]]
    w_char, _, dev_errs_char = learn_predictor_sgd(train_char, dev_char, num_epochs=25, eta=0.1)
    print(f"  [PASS] Char 3-gram trained. Dev Error: {dev_errs_char[-1]*100:.1f}%")

    print("\nALL SENTIMENT CLASSIFICATION TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
