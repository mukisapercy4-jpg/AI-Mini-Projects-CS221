# Theoretical Analysis: Assignment 5 — Pac-Man

**Authors:** Mukisa Percy & Jill  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Multi-Agent Minimax Properties

### 1.1 Definition of the Multi-Agent Game Value
In a zero-sum game with Pac-Man (MAX, agent $0$) and $k$ ghosts (MIN, agents $1, \dots, k$), the game value $V(s)$ at state $s$ is recursively defined as:

$$
V(s) = \begin{cases}
\text{Utility}(s) & \text{if } s \text{ is terminal or depth limit reached} \\
\max_{a \in \text{Actions}(s)} V(\text{Succ}(s, a)) & \text{if agent is Pac-Man (index 0)} \\
\min_{a \in \text{Actions}(s)} V(\text{Succ}(s, a)) & \text{if agent is a ghost (index } 1 \le i \le k)
\end{cases}
$$

### 1.2 Optimality Against Sub-optimal Adversaries
- If the adversary plays sub-optimally (i.e. does not always select $\arg\min V$), the Minimax agent is guaranteed to achieve a score **at least as high** as the Minimax value $V(s)$.
- However, Minimax is pessimistic: it assumes the ghosts are infallible. As a result, it may miss opportunities to exploit foolish ghost moves (which Expectimax can capitalize on).

---

## 2. Alpha-Beta Pruning Efficiency & Cutoff Conditions

### 2.1 Branching Factor and Time Complexity
- Standard Minimax without pruning expands $O(b^d)$ nodes, where $b$ is the branching factor and $d$ is the depth (measured in full rounds of moves).
- With **Alpha-Beta Pruning** and **optimal move ordering** (best moves evaluated first):
  - Leaf evaluations drop to $O(b^{d/2}) = O((\sqrt{b})^d)$.
  - This effectively doubles the searchable depth within the same compute budget.

### 2.2 Pruning Across Multiple Ghosts
For multi-agent Pac-Man with $k$ ghosts, $\alpha$ is maintained as the best (highest) score found so far for MAX, and $\beta$ is the lowest score found so far for MIN.
- When an adversary ghost discovers a move with value $v \le \alpha$, search on that branch terminates immediately (**$\alpha$-cutoff**), because MAX would never allow play to reach this subtree.
- When MAX discovers a move with value $v \ge \beta$, search terminates (**$\beta$-cutoff**), because a higher-level MIN would prevent entering this state.

---

## 3. Expectimax vs. Minimax: Evaluation Invariance

### 3.1 Order-Preserving Transformations
- **Minimax** depends solely on the relative ordering of utility values:
  $$\forall s_1, s_2: u(s_1) > u(s_2) \iff f(u(s_1)) > f(u(s_2))$$
  for any strictly monotonically increasing function $f$. Thus, applying any monotonic transformation to the evaluation function leaves Minimax action decisions unchanged.
- **Expectimax**, however, computes weighted sums (expected values):
  $$E[V] = \sum_{a} P(a) \cdot V(\text{Succ}(s, a))$$
  - While Expectimax is invariant under **positive affine transformations** ($f(x) = ax + b$ with $a > 0$), it is **NOT** invariant under general non-linear monotonic transformations (e.g., $f(x) = x^3$ or $f(x) = \log x$).
  - Non-linear scaling shifts the expected probability-weighted values, potentially causing Expectimax to select different actions.
