# Theoretical Analysis: Assignment 1 — Foundations

**Author:** Henry  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Sparse Vector Representations & Computational Complexity

### 1.1 Time Complexity of Sparse Dot Product
Let $v_1$ and $v_2$ be two sparse vectors represented as hash tables (Python dictionaries) where $\text{nnz}(v_1)$ and $\text{nnz}(v_2)$ denote the number of non-zero entries in each vector respectively.

A naive loop over $v_1$ takes $O(\text{nnz}(v_1))$ operations. By inspecting the lengths and iterating over the smaller dictionary:
$$\text{Cost} = O(\min(\text{nnz}(v_1), \text{nnz}(v_2)))$$
Since hash map lookup $v_2[k]$ is $O(1)$ on average, this is strictly optimal and dominates dense vectors which require $O(D)$ where $D \gg \text{nnz}(v)$ is the vocabulary size.

### 1.2 The Cauchy-Schwarz Inequality in Feature Space
For any two vectors $u, v \in \mathbb{R}^D$:
$$|u \cdot v| \le \|u\|_2 \|v\|_2$$
Equality holds if and only if $u$ and $v$ are linearly dependent ($u = \alpha v$ for scalar $\alpha$). In linear classification and regression, this establishes the geometric margin and normalizes feature projections.

---

## 2. Maximum Likelihood Estimation (MLE)

### 2.1 Bernoulli Distribution
Let $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$, where $X_i \in \{0, 1\}$.
The likelihood function is:
$$L(p) = \prod_{i=1}^n p^{X_i} (1 - p)^{1 - X_i}$$
Taking the log-likelihood:
$$\ell(p) = \sum_{i=1}^n \left[ X_i \log p + (1 - X_i) \log (1 - p) \right] = \left(\sum_{i=1}^n X_i\right) \log p + \left(n - \sum_{i=1}^n X_i\right) \log(1 - p)$$
Taking the derivative with respect to $p$ and setting to 0:
$$\frac{d\ell}{dp} = \frac{\sum_{i=1}^n X_i}{p} - \frac{n - \sum_{i=1}^n X_i}{1 - p} = 0$$
$$(1 - p) \sum_{i=1}^n X_i = p \left(n - \sum_{i=1}^n X_i\right) \implies \sum_{i=1}^n X_i = n p \implies \hat{p}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n X_i$$

### 2.2 Univariate Gaussian Distribution
Let $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$. The log-likelihood is:
$$\ell(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2$$

1. **MLE for Mean $\mu$:**
   $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (X_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n X_i$$
2. **MLE for Variance $\sigma^2$:**
   $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^n (X_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (X_i - \hat{\mu})^2$$

---

## 3. Loss Functions & Gradient Analysis

### 3.1 Ridge Regression Objective
The regularized objective function is:
$$J(w) = \frac{1}{2n} \sum_{i=1}^n (w^T x_i - y_i)^2 + \frac{\lambda}{2} \|w\|_2^2$$
Because both the squared error term and the $L_2$ norm are convex, $J(w)$ is strictly convex when $\lambda > 0$, guaranteeing a unique global minimum.

### 3.2 Gradient Derivation
$$\nabla_w J(w) = \frac{1}{n} \sum_{i=1}^n (w^T x_i - y_i) x_i + \lambda w$$
Gradient descent update with learning rate $\eta$:
$$w^{(t+1)} \leftarrow w^{(t)} - \eta \nabla_w J(w^{(t)}) = (1 - \eta \lambda) w^{(t)} - \frac{\eta}{n} \sum_{i=1}^n (w^{(t)T} x_i - y_i) x_i$$
This demonstrates that $L_2$ regularization operates as weight decay by shrinking weights by a factor of $(1 - \eta \lambda)$ at every gradient step.
