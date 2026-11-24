# Assignment 2: Sentiment Classification

**Assigned Member:** Assigned Member / Open  
**Reference:** [Stanford CS221 Sentiment Analysis Coursework](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

In this assignment, you build a linear binary sentiment classifier (positive $+1$ vs. negative $-1$) trained using Stochastic Gradient Descent (SGD) with Hinge Loss, followed by an unsupervised $K$-Means clustering algorithm on sparse feature vectors:

### 1. Programming Tasks
1. **Feature Extraction (`extract_word_features`)**:
   - Converts arbitrary text reviews into sparse bag-of-words count representations.
   - Case-insensitive tokenization and stop-word filtering.
2. **Stochastic Gradient Descent (SGD) with Hinge Loss**:
   - Linear score: $s(x) = w \cdot \phi(x)$.
   - Margin: $y (w \cdot \phi(x))$.
   - Hinge Loss:
     $$\text{Loss}_{\text{hinge}}(x, y, w) = \max(0, 1 - w \cdot \phi(x) y)$$
   - Subgradient with respect to $w$:
     $$\nabla_w \text{Loss}_{\text{hinge}} = \begin{cases} -y \phi(x) & \text{if } 1 - w \cdot \phi(x) y > 0 \\ 0 & \text{otherwise} \end{cases}$$
   - Weight update on margin violations: $w \leftarrow w + \eta y \phi(x)$.
3. **Evaluation Metrics**:
   - Training error, validation (dev) error, classification accuracy, and margin distributions.
4. **Unsupervised Learning ($K$-Means Clustering)**:
   - Partitioning data points into $K$ cluster centers $\mu_1, \dots, \mu_K$.
   - Iterative assignment and centroid update steps.
   - Tracking the distortion objective:
     $$\mathcal{J}(\mu, z) = \sum_{i=1}^n \|\phi(x_i) - \mu_{z_i}\|_2^2$$

### 2. Theoretical Tasks
1. **Loss Function Analysis**:
   - Comparison of Zero-One loss, Hinge loss, and Logistic loss.
   - Convexity and upper bounding properties of Hinge loss over Zero-One loss.
2. **Subgradient Derivation**:
   - Derivation of subgradients at non-differentiable points (the hinge $1 - y(w \cdot x) = 0$).
3. **$K$-Means Convergence Proof**:
   - Mathematical proof that the $K$-means algorithm is guaranteed to monotonically non-increase the distortion objective $\mathcal{J}$ and converge to a local minimum in a finite number of iterations.

---

## 📁 Module File Structure

- `submission_sentiment.py`: Core algorithms (SGD, Hinge loss, K-Means).
- `sentiment_engine.py`: Movie review dataset generators and tokenization utilities.
- `test_sentiment.py`: Automated unit tests.
- `run_sentiment_experiments.py`: Training runner evaluating accuracy across epochs and cluster distortion.
- `theoretical_sentiment.md`: Theoretical writeup and mathematical proofs.
- `HOW_TO_TEST.md`: Instructions for testing and validation.
