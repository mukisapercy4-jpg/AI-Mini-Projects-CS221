# How to Test — Assignment 2: Sentiment Classification

**Author:** Ssewajje Jolly Hilton (hiltonjolly-2020)
**Module:** `assignments/02_sentiment/`

---

## Quick Start

```bash
cd <repo-root>
python assignments/02_sentiment/test_sentiment.py
```

Expected output:
```
Testing extract_word_features...
  [PASS] extract_word_features: word counts correct.
Testing extract_character_features...
  [PASS] extract_character_features: n-grams correct, spaces stripped.
Testing dot_product...
  [PASS] dot_product: correct sparse inner product.
Testing increment_sparse_vector...
  [PASS] increment_sparse_vector: in-place update correct.
Testing linear prediction (sign of w.phi)...
  [PASS] Linear predictor: sign of score correct.
Testing learn_predictor_sgd (hinge loss)...
  [PASS] SGD trained 25 epochs. Train Error: 0.0%  Dev Error: 0.0%
Testing kmeans_clustering...
  [PASS] K-Means: correct cluster assignments and monotonic distortion.
Testing character n-gram feature performance...
  [PASS] Char 3-gram trained. Dev Error: X.X%

ALL SENTIMENT CLASSIFICATION TESTS PASSED!
```

---

## Test Coverage (8 Test Cases)

| # | Test | Function Tested | Expected |
|---|------|----------------|---------|
| 1 | Word counting | `extract_word_features` | "great" appears twice → `{"great": 2.0, ...}` |
| 2 | Character n-grams + space stripping | `extract_character_features(n)` | "ab cd" → strips space → bigrams `ab, bc, cd` |
| 3 | Sparse dot product | `dot_product` | `{"apple":2,"banana":3} · {"banana":4,"cherry":-1}` = `12.0` |
| 4 | In-place vector update | `increment_sparse_vector` | `d["b"] += 2.0 * 3.0` → `d["b"] == 8.0` |
| 5 | Linear prediction | (local `_predict`) | sign of w·φ: +1 or -1 |
| 6 | SGD hinge-loss training | `learn_predictor_sgd` | Train error < 25% after 25 epochs |
| 7 | K-Means clustering | `kmeans_clustering` | Well-separated clusters; non-increasing distortion |
| 8 | Char n-gram training | `extract_character_features(3)` + `learn_predictor_sgd` | Runs to completion without error |

---

## Run Experiments (Benchmark)

```bash
python assignments/02_sentiment/run_sentiment_experiments.py
```

This trains on a built-in 20-positive / 20-negative review corpus (no external file required) and prints:
- Per-epoch train/dev error for word-unigram features
- Top positive and negative feature weights
- Char 3-gram dev error
- K-Means cluster sizes and distortion per iteration

---

## Module Functions (from `submission_sentiment.py`)

| Function | Signature | Purpose |
|---|---|---|
| `extract_word_features` | `(x: str) → dict` | Word-unigram feature map |
| `extract_character_features` | `(n: int) → Callable` | Character n-gram extractor factory |
| `dot_product` | `(d1, d2) → float` | Sparse inner product |
| `increment_sparse_vector` | `(d1, scale, d2) → None` | In-place: `d1 += scale * d2` |
| `learn_predictor_sgd` | `(train, dev, epochs, eta) → (w, train_errs, dev_errs)` | SGD hinge-loss classifier |
| `learn_predictor` | `(train, dev, feat_fn, epochs, eta) → w` | CS221 official wrapper |
| `kmeans_clustering` | `(points, k, num_iters, seed) → (centroids, assignments, distortions)` | K-Means |
| `kmeans` | `(examples, K, max_epochs) → (centroids, assignments, loss)` | CS221 notebook wrapper |

---

## Notes

- **No external dataset** required — both the test suite and the experiments runner use a built-in corpus.
- Functions follow **CS221 official names** (`learn_predictor`, `kmeans`) in the master notebook and
  `submission_sentiment.py`-style names (`learn_predictor_sgd`, `kmeans_clustering`) in the helper runner.
- Requires only Python standard library (`random`, `collections`).
