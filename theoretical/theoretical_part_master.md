# MAKERERE UNIVERSITY
## DEPARTMENT OF COMPUTER SCIENCE
### Artificial Intelligence Mini-Project Assignments (CS221 Aligned)
**Theoretical Solutions Master Document**

---

### Course & Submission Metadata
- **Course Portal:** [MUELE Section 230921](https://muele.mak.ac.ug/course/section.php?id=230921)
- **Reference Coursework:** [Stanford CS221: Principles and Techniques](https://stanford-cs221.github.io/spring2026/#coursework)
- **Submission Deadline:** 5th December 2026
- **Format:** Single Unified PDF Document

### Team Members & Contributions
| # | Full Name | Student / Reg No. | Email | Assigned Theoretical Section |
|---|-----------|-------------------|-------|------------------------------|
| 1 | Henry | [Reg No] | [Email] | Section 1: Foundations |
| 2 | [Member Name] | [Reg No] | [Email] | Section 2: Sentiment Classification |
| 3 | Precious Atwiine | [Reg No] | [Email] | Section 3: Route Planning |
| 4 | Marcus | [Reg No] | [Email] | Section 4: Controlling MountainCar |
| 5 | [Co-Lead] | [Reg No] | [Email] | Section 4: Controlling MountainCar |
| 6 | Jill | [Reg No] | [Email] | Section 5: Pac-Man |
| 7 | Mukisa Percy | [Reg No] | mukisapercy4@gmail.com | Section 5: Pac-Man |
| 8 | Joseph Mary | [Reg No] | [Email] | Section 6: Course Scheduling / Bayesian |
| 9 | Josh Kwizera | [Reg No] | [Email] | Section 7: Car Tracking / Logic |
| 10 | [Integration Lead] | [Reg No] | [Email] | Section 8: Compilation & Review |

---

## Section 1: Foundations (`foundations`)
**Author:** Henry

### 1.1 Mathematical Optimization & Vector Spaces
- **Problem Formulation:**
- **Derivations:**
- **Complexity Analysis:**

### 1.2 Probability Foundations & Maximum Likelihood
- **Derivations:**
- **Proofs:**

---

## Section 2: Sentiment Classification (`sentiment`)
**Author:** [Assigned Member]

### 2.1 Loss Functions & Gradient Analysis
- **Hinge Loss vs. Logistic Loss:**
- **Sub-gradient Derivation:**

### 2.2 Unsupervised Learning (K-Means Clustering)
- **Convergence Guarantees:**
- **Objective Function Optimization:**

---

## Section 3: Route Planning (`route`)
**Author:** Precious Atwiine

### 3.1 Admissibility and Consistency of Heuristics
- **Definitions & Mathematical Proof:**
- **Landmark Heuristic Construction:**

### 3.2 Algorithm Comparison: Dijkstra vs. A*
- **Search Space Expansion:**
- **Optimality Trade-offs:**

---

## Section 4: Controlling MountainCar (`mountaincar`)
**Authors:** Marcus & Co-Lead

### 4.1 Markov Decision Process (MDP) Formulation
- **State Space, Action Space, Transition Dynamics:**
- **Bellman Optimality Equations:**

### 4.2 Value Iteration vs. Policy Iteration vs. Q-Learning
- **Contraction Mapping Proof Outline:**
- **Convergence Properties under Function Approximation:**

---

## Section 5: Pac-Man Multi-Agent Adversarial Search (`pacman`)
**Authors:** Mukisa Percy & Jill

### 5.1 Multi-Agent Minimax Game Tree Formulation
In the Pac-Man environment, games are zero-sum with one maximizing agent (Pac-Man, index $0$) and $k$ minimizing adversaries (Ghosts, indices $1, \dots, k$). One search ply corresponds to a sequence of $1 + k$ agent moves.

The game value $V(s)$ at state $s$ is defined by the generalized recurrence:
$$
V(s) = \begin{cases}
\text{Utility}(s), & \text{if } s \text{ is terminal or } \text{depth} = d_{\max} \\
\max_{a \in \text{Actions}(s, 0)} V(\text{Succ}(s, a)), & \text{if agent is Pac-Man (index 0)} \\
\min_{a \in \text{Actions}(s, i)} V(\text{Succ}(s, a)), & \text{if agent is ghost } i \in \{1, \dots, k\}
\end{cases}
$$

#### Optimality Against Sub-Optimal Adversaries:
- **Pessimism Guarantee:** If ghosts do not play optimally (e.g. they choose actions randomly or follow heuristic sub-optimal policies), the actual achieved utility of the Minimax agent is guaranteed to be **at least as high** as the Minimax theoretical value:
  $$\text{Score}_{\text{actual}} \ge V_{\text{minimax}}(s_0)$$
- **Opportunity Cost:** While safe against death, Minimax cannot actively exploit non-optimal behavior (such as trapping a wandering ghost) because it always assumes the ghost will make the deadliest possible counter-move.

---

### 5.2 Alpha-Beta Pruning Correctness & Cutoff Criteria
Alpha-Beta pruning eliminates subtrees that cannot influence the final root decision, preserving the exact Minimax decision:

$$\alpha = \max \text{ value found so far for MAX across the path to root}$$
$$\beta = \min \text{ value found so far for MIN across the path to root}$$

#### Cutoff Conditions in Multi-Agent Ghost Trees:
1. **$\beta$-Pruning at MAX Node (Pac-Man):**
   When examining action $a$ at a MAX node, if the returned child value $v$ satisfies:
   $$v \ge \beta$$
   the search truncates immediately. An ancestor MIN node already has an alternative path with value $\beta \le v$, meaning MAX would never be permitted to reach this subtree.

2. **$\alpha$-Pruning at MIN Node (Ghost $i$):**
   When examining action $a$ at any ghost MIN node, if the returned value satisfies:
   $$v \le \alpha$$
   the search truncates immediately. The root MAX node already has a guaranteed path yielding $\alpha \ge v$, meaning MIN can already force a score lower than what MAX would accept.

#### Computational Complexity with Optimal Move Ordering:
- **Standard Minimax:** $O(b^d)$ node evaluations, where $b$ is the average branching factor and $d$ is depth.
- **Optimal Ordering Alpha-Beta:** By visiting the highest-utility actions first for MAX and lowest-utility actions first for MIN, the effective branching factor is halved:
  $$T(d) = O\left(b^{\lceil d/2 \rceil} + b^{\lfloor d/2 \rfloor}\right) = O\left(b^{d/2}\right) = O\left((\sqrt{b})^d\right)$$
  This allows searching twice as deep within the same CPU time limit.

---

### 5.3 Expectimax vs. Minimax: Invariance Properties Under Transformations

#### 1. Strictly Monotonic Non-Linear Transformations:
Let $f: \mathbb{R} \to \mathbb{R}$ be a strictly monotonically increasing function ($u > v \iff f(u) > f(v)$).
- **Minimax Invariance:** Minimax decisions depend solely on the $\arg\max$ and $\arg\min$ orderings:
  $$\arg\max_{a} f(V(s_a)) = \arg\max_{a} V(s_a)$$
  Therefore, Minimax policies are completely invariant under any arbitrary monotonic transformation of the evaluation function.
- **Expectimax Non-Invariance:** Expectimax computes expected values over probability distributions $P(a)$:
  $$\mathbb{E}[V] = \sum_{a} P(a) V(s_a) \quad \text{vs.} \quad \sum_{a} P(a) f(V(s_a))$$
  By Jensen's Inequality, for non-linear functions (e.g. $f(x) = x^3$ or $f(x) = e^x$), the relative rank of expectations is not preserved:
  $$\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$$
  Consequently, Expectimax may alter its chosen actions when the evaluation function is subjected to non-linear scaling.

#### 2. Positive Affine Transformations ($f(x) = c \cdot x + d, c > 0$):
Under positive affine transformations, expectation is strictly linear:
$$\mathbb{E}[c \cdot X + d] = c \cdot \mathbb{E}[X] + d$$
Because $c > 0$, the $\arg\max$ remains unchanged. Hence, Expectimax is invariant **only under positive affine transformations**.

---

### 5.4 State Evaluation Function Formulation
Our `betterEvaluationFunction` evaluates game states $s$ via a linear combination of domain features:
$$\text{Eval}(s) = w_1 \cdot \text{Score}(s) + w_2 \cdot \frac{1}{d_{\text{food}}(s) + 1} - w_3 \cdot N_{\text{food}}(s) - w_4 \cdot N_{\text{capsules}}(s) + \sum_{g \in \text{Ghosts}} \text{GhostTerm}(g, s)$$
where:
$$\text{GhostTerm}(g, s) = \begin{cases}
-\infty, & \text{if } \text{Dist}(P, g) \le 1 \text{ and } \text{Scared}(g) = 0 \\
-\frac{3.0}{\text{Dist}(P, g) + 0.1}, & \text{if } \text{Scared}(g) = 0 \\
+\frac{10.0}{\text{Dist}(P, g) + 0.1}, & \text{if } \text{Scared}(g) > 0
\end{cases}$$
This balances cautious survival against active ghosts with aggressive pursuit during power-pellet activation.

---

## Section 6: Course Scheduling & Bayesian Networks (`scheduling` / `bayesian`)
**Author:** Joseph Mary

### 6.1 Constraint Satisfaction Problems (CSP)
- **Constraint Graph Formulation:**
- **Arc Consistency (AC-3) & Forward Checking Complexity:**

### 6.2 Bayesian Networks & Conditional Independence
- **D-separation Queries:**
- **Exact Inference (Variable Elimination) vs. Approximate Sampling:**

---

## Section 7: Car Tracking & From Language to Logic (`tracking` / `logic`)
**Author:** Josh Kwizera

### 7.1 Hidden Markov Models (HMM) & Particle Filtering
- **Forward-Backward Algorithm Derivation:**
- **Particle Depletion & Importance Resampling:**

### 7.2 Propositional & First-Order Logic
- **Conjunctive Normal Form (CNF) Transformation:**
- **Resolution Proofs and Soundness/Completeness:**
