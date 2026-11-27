# Theoretical Analysis: Assignment 6 — Course Scheduling & Bayesian Networks

**Author:** Joseph Mary Yiga (Student No. 2500703615)  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Constraint Satisfaction Problems (CSP)

### 1.1 Constraint Graph Formulation
A CSP is a tuple $(X, D, C)$ where:
- $X = \{X_1, \dots, X_n\}$ is the set of variables (e.g., courses to schedule).
- $D = \{D_1, \dots, D_n\}$ is the set of finite domains (e.g., offered quarters $\{1, 2, 3, 4\}$).
- $C$ is a set of factors/constraints $f_j(x) \ge 0$ restricting simultaneous assignments (e.g., $\text{Quarter}(\text{CS101}) < \text{Quarter}(\text{CS102})$).

### 1.2 Arc Consistency (AC-3) Complexity
Let $n = |X|$ be the number of variables, $e$ the number of binary constraints (edges), and $d = \max_i |D_i|$ the maximum domain size:
- Each directed arc $(X_i, X_j)$ can be re-enqueued at most $d$ times (only when $D_i$ shrinks by at least one element).
- Checking arc $(X_i, X_j)$ via `revise` takes $O(d^2)$ pairs.
- With $2e$ directed arcs, the worst-case time complexity of **AC-3** is:
  $$O(e \cdot d^3)$$

### 1.3 Heuristics: MRV & LCV
- **Minimum Remaining Values (MRV / Fail-First):** Select the unassigned variable with the smallest remaining domain $|D_i|$, pruning dead-end subtrees as early as possible.
- **Least Constraining Value (LCV / Fail-Last):** Order values to rule out the fewest choices for neighboring unassigned variables, maximizing the chance of finding a valid completion quickly.

---

## 2. Bayesian Networks & Probabilistic Inference

### 2.1 Joint Distribution Factorization
A Bayesian Network represents a joint distribution over variables $X_1, \dots, X_n$ via a DAG:
$$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Parents}(X_i))$$

### 2.2 D-Separation Rules (Conditional Independence)
A path between $A$ and $B$ is **blocked** by evidence set $E$ if it contains a triple $(X, Y, Z)$ where:
1. **Chain** ($X \to Y \to Z$) or **Fork** ($X \leftarrow Y \to Z$): $Y \in E$ (observing the middle node blocks flow).
2. **Collider / V-structure** ($X \to Y \leftarrow Z$): neither $Y$ nor any descendant of $Y$ is in $E$ (observing a collider *opens* the path — **explaining away**).

### 2.3 Exact Inference vs. Approximate Sampling
- **Exact Inference (Enumeration / Variable Elimination):** Exact posteriors, but worst-case complexity is $O(n \cdot d^{w+1})$ exponential in the graph's induced treewidth $w$.
- **Likelihood Weighting:** Fixes evidence variables $E$, samples non-evidence variables forward in topological order, and weights each sample by $\prod_{e \in E} P(e \mid \text{Parents}(e))$. Scales linearly per sample and converges as $O(1/\sqrt{N})$.
