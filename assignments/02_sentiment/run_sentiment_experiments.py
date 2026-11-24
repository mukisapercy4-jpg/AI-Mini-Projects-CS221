"""
Sentiment Experiment & Benchmark Runner
Trains sentiment predictor over multiple epochs and visualizes cluster distortion decay.
"""

import time
from submission_sentiment import (
    extract_word_features,
    learn_predictor_sgd,
    kmeans_clustering
)
from sentiment_engine import get_movie_review_dataset

def run_experiments():
    print("=" * 60)
    print("CS221 ASSIGNMENT 2: SENTIMENT EXPERIMENTS")
    print("=" * 60)

    # 1. Supervised Sentiment Classification
    raw_train, raw_dev = get_movie_review_dataset(num_train=50, num_dev=20, seed=42)
    train_data = [(extract_word_features(text), y) for text, y in raw_train]
    dev_data = [(extract_word_features(text), y) for text, y in raw_dev]

    print(f"[DATASET] Train size: {len(train_data)} reviews | Dev size: {len(dev_data)} reviews")
    
    t0 = time.perf_counter()
    w, train_errs, dev_errs = learn_predictor_sgd(
        train_data, dev_data, num_epochs=15, eta=0.05
    )
    train_time = (time.perf_counter() - t0) * 1000.0

    print(f"[PERFORMANCE] Trained 15 epochs in {train_time:.2f} ms")
    print("\nEpoch | Train Error | Dev Error | Top Positive / Negative Feature Weights")
    print("-" * 65)
    
    sorted_weights = sorted(w.items(), key=lambda item: item[1], reverse=True)
    top_pos = [k for k, v in sorted_weights[:3]]
    top_neg = [k for k, v in sorted_weights[-3:]]
    
    for epoch in range(0, 15, 3):
        print(f"{epoch+1:5d} | {train_errs[epoch]*100:10.1f}% | {dev_errs[epoch]*100:8.1f}% | Pos: {top_pos} | Neg: {top_neg}")

    # 2. Unsupervised K-Means Clustering Benchmark
    print("\n" + "=" * 60)
    print("[UNSUPERVISED] Running K-Means Clustering on Review Features (K=2):")
    feature_points = [phi for phi, _ in train_data]
    
    t_km = time.perf_counter()
    centroids, assignments, distortions = kmeans_clustering(feature_points, k=2, num_iters=10, seed=42)
    km_time = (time.perf_counter() - t_km) * 1000.0

    print(f"K-Means converged in {km_time:.2f} ms")
    print("Iteration | Total Distortion J(mu, z)")
    print("-" * 35)
    for it, dist in enumerate(distortions[:6]):
        print(f"{it+1:9d} | {dist:15.4f}")

    print("=" * 60)
    print("Sentiment experiments completed successfully.")

if __name__ == "__main__":
    run_experiments()
