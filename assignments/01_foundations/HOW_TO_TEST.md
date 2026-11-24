# How to Test Assignment 1: Foundations

**Assigned Reviewer:** Henry & Integration Lead  
**Folder:** `assignments/01_foundations/`

---

## 🧪 1. Quick Automated Unit Tests
Run the standalone unit tests:
```powershell
python assignments/01_foundations/test_foundations.py
```
Expected output:
```text
Testing Sparse Vector Dot Product...
  [PASS] dot_product verified.
Testing In-Place Sparse Vector Increment...
  [PASS] increment_sparse_vector verified.
Testing Word Frequency and Longest Word...
  [PASS] Word processing verified.
Testing Bernoulli and Gaussian MLE...
  [PASS] MLE derivations verified.
Testing Ridge Regression Gradient Descent...
  [PASS] Model trained. Train MSE: ...
ALL BASIC FOUNDATIONS TESTS PASSED!
```

---

## 📊 2. Performance Benchmarks
Benchmark sparse vector operations and gradient descent iterations:
```powershell
python assignments/01_foundations/run_foundations_experiments.py
```

---

## 📄 3. Theoretical Verification
- Review `theoretical_foundations.md`.
- Verify the mathematical derivations for:
  - Time complexity of sparse vector dot product: $O(\min(|d_1|, |d_2|))$.
  - Maximum Likelihood Estimation of Bernoulli $\hat{p}$ and Gaussian $(\hat{\mu}, \hat{\sigma}^2)$.
  - Analytical gradient of Ridge Regression with weight decay interpretation.
