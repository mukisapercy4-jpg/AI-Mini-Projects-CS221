# Theoretical Analysis: Assignment 2 — Sentiment Classification

**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)  
**Topic:** Linear Classifiers, Hinge Loss, Subgradients & K-Means Clustering

---

## 1. Linear Predictor & Loss Function Comparison

### 1.1 The Linear Margin
For input feature vector $\phi(x) \in \mathbb{R}^D$, weight vector $w \in \mathbb{R}^D$, and true label $y \in \{-1, +1\}$, the margin is defined as:
$$m(x, y, w) = (w \cdot \phi(x)) \cdot y$$
- $m > 0 \implies$ Correct prediction.
- $m < 0 \implies$ Incorrect prediction.
- $|m| \implies$ Confidence of the prediction.

### 1.2 Comparison of Common Loss Functions
1. **Zero-One Loss:**
   $$\text{Loss}_{0-1}(x, y, w) = \mathbf{1}[m(x, y, w) \le 0]$$
   - Discontinuous, non-convex, and NP-hard to optimize directly.
2. **Hinge Loss:**
   $$\text{Loss}_{\text{hinge}}(x, y, w) = \max(0, 1 - m(x, y, w)) = \max(0, 1 - (w \cdot \phi(x)) y)$$
   - Convex upper bound on the zero-one loss.
   - Demands a "margin of safety" ($m \ge 1$), penalizing both incorrect classifications and marginally correct ones ($0 < m < 1$).
3. **Logistic Loss:**
   $$\text{Loss}_{\text{logistic}}(x, y, w) = \log(1 + e^{-m(x, y, w)})$$
   - Smooth, strictly convex everywhere, and approaches zero asymptotically.

---

## 2. Subgradient Derivation of Hinge Loss

Because the hinge function $f(z) = \max(0, 1 - z)$ contains a non-differentiable kink at $z = 1$, we employ the subgradient:
$$\partial_w \text{Loss}_{\text{hinge}}(x, y, w) = \begin{cases}
-y \phi(x) & \text{if } 1 - (w \cdot \phi(x)) y > 0 \quad (\text{margin violation}) \\
0 & \text{if } 1 - (w \cdot \phi(x)) y < 0 \quad (\text{safe margin}) \\
\{ -c \cdot y \phi(x) \mid c \in [0, 1] \} & \text{if } 1 - (w \cdot \phi(x)) y = 0 \quad (\text{subdifferential})
\end{cases}$$

### Stochastic Gradient Descent (SGD) Update Rule:
For learning rate $\eta > 0$:
$$w^{(t+1)} \leftarrow \begin{cases}
w^{(t)} + \eta y \phi(x) & \text{if } (w^{(t)} \cdot \phi(x)) y < 1 \\
w^{(t)} & \text{otherwise}
\end{cases}$$

---

## 3. Mathematical Proof of $K$-Means Convergence

The $K$-Means distortion objective is defined as:
$$\mathcal{J}(\mu, z) = \sum_{i=1}^n \|\phi(x_i) - \mu_{z_i}\|_2^2$$
where $z_i \in \{1, \dots, K\}$ is the cluster assignment for data point $i$, and $\mu_k \in \mathbb{R}^D$ is the center of cluster $k$.

### Proof Steps:
1. **Assignment Step (Minimizing over $z$ with fixed $\mu$):**
   $$z_i^{(t)} = \arg\min_{k \in \{1, \dots, K\}} \|\phi(x_i) - \mu_k^{(t-1)}\|_2^2$$
   By construction, choosing the minimum Euclidean distance for each $i$ independently strictly minimizes $\mathcal{J}$ holding $\mu$ constant:
   $$\mathcal{J}(\mu^{(t-1)}, z^{(t)}) \le \mathcal{J}(\mu^{(t-1)}, z^{(t-1)})$$

2. **Centroid Update Step (Minimizing over $\mu$ with fixed $z$):**
   Holding assignments $z$ fixed, $\mathcal{J}$ decomposes into $K$ independent sub-problems. Setting the gradient with respect to $\mu_k$ to zero:
   $$\nabla_{\mu_k} \sum_{i: z_i = k} \|\phi(x_i) - \mu_k\|_2^2 = -2 \sum_{i: z_i = k} (\phi(x_i) - \mu_k) = 0 \implies \mu_k^{(t)} = \frac{1}{|C_k|} \sum_{i: z_i = k} \phi(x_i)$$
   Since the squared Euclidean norm is strictly convex, the sample mean is the unique global minimizer for each cluster:
   $$\mathcal{J}(\mu^{(t)}, z^{(t)}) \le \mathcal{J}(\mu^{(t-1)}, z^{(t)})$$

3. **Monotonicity and Finite Termination:**
   Combining both steps:
   $$\mathcal{J}(\mu^{(t)}, z^{(t)}) \le \mathcal{J}(\mu^{(t-1)}, z^{(t-1)})$$
   Because the objective is bounded below ($\mathcal{J} \ge 0$) and the number of discrete cluster assignments is finite ($K^n$), the algorithm must terminate at a local minimum in a finite number of iterations without cycling. $\blacksquare$
