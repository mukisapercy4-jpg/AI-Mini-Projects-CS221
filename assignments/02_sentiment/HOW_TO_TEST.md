# How to Test Assignment 2: Sentiment Classification

**Assigned Reviewer:** Reviewer & Integration Lead  
**Folder:** `assignments/02_sentiment/`

---

## 🧪 1. Quick Automated Unit Tests
Run the standalone unit tests:
```powershell
python assignments/02_sentiment/test_sentiment.py
```
Expected output:
```text
Testing Feature Extraction...
  [PASS] extract_word_features verified.
Testing Linear Predictor...
  [PASS] predict_linear verified.
Testing Sentiment SGD Classifier...
  [PASS] SGD trained. Final Train Error: ...
Testing K-Means Clustering...
  [PASS] K-Means clustering verified.
ALL BASIC SENTIMENT TESTS PASSED!
```

---

## 📊 2. Performance & Convergence Experiments
Run the training curves and clustering distortion diagnostics:
```powershell
python assignments/02_sentiment/run_sentiment_experiments.py
```

---

## 📄 3. Theoretical Verification
- Review `theoretical_sentiment.md`.
- Verify the mathematical derivations for:
  - Zero-One vs. Hinge vs. Logistic loss bounds.
  - Subgradient of Hinge loss with respect to $w$.
  - Two-step coordinate descent proof of $K$-Means monotonic convergence.
