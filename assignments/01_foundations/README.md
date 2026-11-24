# Assignment 1: Foundations (CS221 Warmup & Mathematics)

**Assigned Member:** Henry  
**Reference:** [Stanford CS221 Foundations Coursework](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

This assignment covers foundational Python skills, efficient sparse vector operations, optimization, and probability derivations required throughout CS221:

### 1. Programming Tasks
1. **Sparse Vector Operations (`submission_foundations.py`)**:
   - `dot_product(d1, d2)`: Computes the dot product of two sparse vectors represented as dictionaries in $O(\min(|d1|, |d2|))$ time.
   - `increment_sparse_vector(d1, scale, d2)`: In-place addition $d1 \leftarrow d1 + \text{scale} \times d2$.
2. **Text Processing & Feature Extraction**:
   - `compute_word_frequencies(sentence)`: Normalizes and tokenizes text, returning word counts.
   - `find_longest_word(text)`: Identifies the longest token by character length.
3. **Loss Function Optimization (Ridge Regression with Gradient Descent)**:
   - Squared loss with $L_2$ regularization:
     $$J(w) = \frac{1}{2n} \sum_{i=1}^n (w \cdot x_i - y_i)^2 + \frac{\lambda}{2} \|w\|_2^2$$
   - Gradient computation:
     $$\nabla_w J(w) = \frac{1}{n} \sum_{i=1}^n (w \cdot x_i - y_i) x_i + \lambda w$$
   - `gradient_descent(points, num_iters, eta, reg_lambda)`: Optimizes sparse weight vector.

### 2. Theoretical Tasks
1. **Mathematical Optimization & Cauchy-Schwarz Inequality**:
   - Analysis of inner products and projection in high-dimensional feature spaces.
2. **Maximum Likelihood Estimation (MLE)**:
   - Analytical derivation of MLE parameter $\hat{p}$ for a coin tossing experiment under the Bernoulli distribution.
   - Derivation of MLE mean $\hat{\mu}$ and variance $\hat{\sigma}^2$ for the Gaussian distribution.
3. **Complexity Guarantees**:
   - Time and space complexity proofs for sparse vector computations versus dense representations.

---

## 📁 Module File Structure

- `submission_foundations.py`: Core algorithm implementations.
- `foundations_engine.py`: Synthetic benchmark dataset generators and mathematical validators.
- `test_foundations.py`: Automated unit tests verifying mathematical correctness and edge cases.
- `run_foundations_experiments.py`: Benchmark runner evaluating runtime efficiency and convergence rates.
- `theoretical_foundations.md`: Complete theoretical writeup and formal proofs.
- `HOW_TO_TEST.md`: Instructions for Henry and team reviewers.
