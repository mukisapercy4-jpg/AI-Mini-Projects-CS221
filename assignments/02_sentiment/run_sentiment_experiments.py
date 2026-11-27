"""
Sentiment Experiment & Benchmark Runner — Assignment 2 (CS221)
Author: Ssewajje Jolly Hilton (hiltonjolly-2020)

Trains a sentiment predictor with SGD over multiple epochs,
evaluates word-unigram vs character n-gram features, and
demonstrates K-Means clustering on feature vectors.

No external dataset required — uses a built-in review corpus.
"""

import time
import random
from submission_sentiment import (
    extract_word_features,
    extract_character_features,
    dot_product,
    learn_predictor_sgd,
    kmeans_clustering,
)

# ---------------------------------------------------------------------------
# Built-in mini corpus (no external file dependency)
# ---------------------------------------------------------------------------
_POS = [
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
    "deeply moving and beautifully crafted",
    "remarkable film with stellar performances",
    "enchanting story with vivid characters",
    "joyful uplifting and wonderfully acted",
    "a triumph of storytelling and direction",
    "inspired filmmaking that lingers in the mind",
    "radiant performances and exquisite cinematography",
    "a genuinely moving and memorable experience",
    "gloriously funny and emotionally resonant",
    "ambitious and brilliantly realised",
]
_NEG = [
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
    "uninspired and tediously predictable throughout",
    "laughably bad acting ruins every scene",
    "forgettable cliched and deeply disappointing",
    "dreadful misfire that wastes every asset",
    "appalling waste of talent and budget",
    "incoherent plot and excruciatingly dull pacing",
    "poor production values and wooden performances",
    "deeply flawed uninspiring and badly edited",
    "derivative nonsense with zero emotional impact",
    "a dreary joyless slog from start to finish",
]


def _build_dataset(seed=42, dev_ratio=0.25):
    random.seed(seed)
    raw = [(r, 1) for r in _POS] + [(r, -1) for r in _NEG]
    random.shuffle(raw)
    split = max(1, int(len(raw) * (1 - dev_ratio)))
    return raw[:split], raw[split:]


def run_experiments():
    print("=" * 60)
    print("CS221 ASSIGNMENT 2: SENTIMENT EXPERIMENTS")
    print("=" * 60)

    raw_train, raw_dev = _build_dataset()

    # ── 1. Word-Unigram Features ───────────────────────────────────────────
    train_data = [(extract_word_features(x), y) for x, y in raw_train]
    dev_data   = [(extract_word_features(x), y) for x, y in raw_dev]

    print(f"[DATASET] Train: {len(train_data)} reviews | Dev: {len(dev_data)} reviews")

    t0 = time.perf_counter()
    w, train_errs, dev_errs = learn_predictor_sgd(
        train_data, dev_data, num_epochs=15, eta=0.05
    )
    train_time = (time.perf_counter() - t0) * 1000.0

    print(f"[PERFORMANCE] Word features — 15 epochs in {train_time:.2f} ms")
    print("\nEpoch | Train Error | Dev Error")
    print("-" * 35)
    for epoch in range(0, 15, 3):
        print(f"{epoch+1:5d} | {train_errs[epoch]*100:10.1f}% | {dev_errs[epoch]*100:8.1f}%")

    sorted_w = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nTop 5 positive features: {[k for k, v in sorted_w[:5]]}")
    print(f"Top 5 negative features: {[k for k, v in sorted_w[-5:]]}")

    # ── 2. Character 3-gram Features ──────────────────────────────────────
    ext3 = extract_character_features(3)
    train_char = [(ext3(x), y) for x, y in raw_train]
    dev_char   = [(ext3(x), y) for x, y in raw_dev]

    t1 = time.perf_counter()
    w_char, _, dev_errs_char = learn_predictor_sgd(
        train_char, dev_char, num_epochs=15, eta=0.05
    )
    char_time = (time.perf_counter() - t1) * 1000.0

    print(f"\n[PERFORMANCE] Char 3-gram — 15 epochs in {char_time:.2f} ms")
    print(f"  Final Dev Error (char 3-gram): {dev_errs_char[-1]*100:.1f}%")

    # ── 3. K-Means Clustering ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[UNSUPERVISED] K-Means Clustering (K=2) on Review Features:")
    feature_points = [phi for phi, _ in train_data]

    t_km = time.perf_counter()
    centroids, assignments, distortions = kmeans_clustering(
        feature_points, k=2, num_iters=10, seed=42
    )
    km_time = (time.perf_counter() - t_km) * 1000.0

    c0 = sum(1 for a in assignments if a == 0)
    c1 = sum(1 for a in assignments if a == 1)
    print(f"K-Means converged in {km_time:.2f} ms")
    print(f"Cluster sizes: {c0} and {c1}")
    print("Iteration | Total Distortion J(mu, z)")
    print("-" * 35)
    for it, dist in enumerate(distortions[:6]):
        print(f"{it+1:9d} | {dist:15.4f}")

    print("=" * 60)
    print("Sentiment experiments completed successfully.")


if __name__ == "__main__":
    run_experiments()
