# Theoretical Analysis: Assignment 3 — Route Planning

**Author:** Precious Atwiine  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Heuristic Properties: Admissibility and Consistency

### 1.1 Mathematical Definitions
Let $s$ be any state in a state-space graph, $g \in \text{Goals}$ a goal state, and $h^*(s)$ the exact optimal cost from $s$ to $g$.

1. **Admissibility:**
   A heuristic $h(s)$ is admissible if it never overestimates the true cost to reach the nearest goal:
   $$\forall s, \quad 0 \le h(s) \le h^*(s)$$
   with $h(g) = 0$ for all $g \in \text{Goals}$.

2. **Consistency (Monotonicity):**
   A heuristic $h(s)$ is consistent if for every state $s$ and successor $s'$ reached via action $a$ with non-negative edge cost $c(s, a, s')$:
   $$h(s) \le c(s, a, s') + h(s')$$
   and $h(g) = 0$.

### 1.2 Theorem: Consistency Implies Admissibility
**Proof by Induction:**  
Let $s = s_0, s_1, s_2, \dots, s_k = g$ denote the sequence of states along the true optimal path from $s$ to $g$. By definition, the optimal cost is:
$$h^*(s) = \sum_{i=0}^{k-1} c(s_i, a_i, s_{i+1})$$
Applying the consistency condition recursively:
$$h(s_0) \le c(s_0, a_0, s_1) + h(s_1)$$
$$h(s_1) \le c(s_1, a_1, s_2) + h(s_2)$$
$$\dots$$
$$h(s_{k-1}) \le c(s_{k-1}, a_{k-1}, g) + h(g)$$
Summing these $k$ inequalities and telescoping:
$$h(s_0) \le \sum_{i=0}^{k-1} c(s_i, a_i, s_{i+1}) + h(g) = h^*(s_0) + 0 = h^*(s)$$
Hence, consistency strictly implies admissibility. $\blacksquare$

---

## 2. Landmark (Differential) Heuristic Construction & Proof

Let $L$ be a designated landmark node in an undirected (or symmetric) road network, and let $d(u, v)$ be the true shortest path distance between any two nodes $u$ and $v$.
The landmark heuristic with respect to goal $g$ is defined as:
$$h_L(s) = |d(s, L) - d(g, L)|$$

### 2.1 Proof of Consistency:
We must show that for any edge $(u, v)$ with cost $c(u, v)$:
$$h_L(u) \le c(u, v) + h_L(v)$$
By the triangle inequality of the metric space defined by shortest paths:
1. $d(u, L) \le d(u, v) + d(v, L) \le c(u, v) + d(v, L) \implies d(u, L) - d(v, L) \le c(u, v)$
2. $d(v, L) \le d(v, u) + d(u, L) \le c(u, v) + d(u, L) \implies d(v, L) - d(u, L) \le c(u, v)$
Combining both: $|d(u, L) - d(v, L)| \le c(u, v)$.

Now, consider $h_L(u) - h_L(v)$:
$$h_L(u) - h_L(v) = |d(u, L) - d(g, L)| - |d(v, L) - d(g, L)|$$
Applying the reverse triangle inequality for absolute values $|a| - |b| \le |a - b|$:
$$h_L(u) - h_L(v) \le |(d(u, L) - d(g, L)) - (d(v, L) - d(g, L))| = |d(u, L) - d(v, L)| \le c(u, v)$$
$$\implies h_L(u) \le c(u, v) + h_L(v)$$
Furthermore, at the goal state:
$$h_L(g) = |d(g, L) - d(g, L)| = 0$$
Thus, $h_L(s)$ is consistent everywhere, and therefore admissible. $\blacksquare$

---

## 3. Optimality & Expansion Properties: UCS vs. A*

- **Uniform Cost Search (UCS):** Expands nodes in concentric contours of $g(s)$, effectively conducting an undirected circular wavefront search.
- **A* Search:** Guides the search frontier along an ellipse directed toward the goal.
- **Optimality Guarantee with Consistent Heuristic:** A* with a consistent heuristic is guaranteed to find an optimal path without ever needing to re-open closed nodes, exploring a subset of the nodes expanded by UCS.
