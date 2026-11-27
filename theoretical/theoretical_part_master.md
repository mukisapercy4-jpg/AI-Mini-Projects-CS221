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
| 1 | Ssenyondo Henry | 2500703882 | ssenyondohenry644@gmail.com | Section 1: Foundations |
| 2 | Ssewajje Jolly Hilton | 2500703588 / 25/U/03588/EVE | hiltonjolly@gmail.com | Section 2: Sentiment Classification |
| 3 | Atwiine Precious Mwebaze | 2500703347 | preciousinshuti3@gmail.com | Section 3: Route Planning |
| 4 | Opolot Emmanuel Marcus | 2500729501 | marcusemma480@gmail.com | Section 4: Controlling MountainCar |
| 5 | Kabanda Joel | 2500703381 | joelkabanda45@gmail.com | Section 4: Controlling MountainCar |
| 6 | Munanura Jill | 2500705824 | jillmunanura32@gmail.com | Section 5: Pac-Man |
| 7 | Mukisa Percy | 2500703468 / 25/U/03468/EVE | mukisapercy4@gmail.com | Section 5: Pac-Man |
| 8 | Joseph Mary Yiga | 2500703615 | josephmaryyiga@gmail.com | Section 6: Course Scheduling / Bayesian |
| 9 | Muzira Joshua Kwizera | 2500703484 / 25/U/03484/EVE | joshngoboka1@gmail.com | Section 7: Car Tracking / Logic |
| 10 | Ariko Joseph Ojangole | 2500703330 / 25/U/03330/EVE | arikolojangole@gmail.com | Section 8: Integration & Submission Lead |

---

## Section 1: Foundations (`foundations`)
**Author:** Ssenyondo Henry (Student No. 2500703882)

### 1.1 Mathematical Optimization & Vector Spaces
- **Sparse Dot Product Complexity:** Representing sparse vectors as hash tables with $\text{nnz}(v_1), \text{nnz}(v_2)$ entries and iterating over the smaller dictionary gives optimal $O(\min(\text{nnz}(v_1), \text{nnz}(v_2)))$ runtime, dominating $O(D)$ dense vectors.
- **Cauchy-Schwarz Inequality:** $|u \cdot v| \le \|u\|_2 \|v\|_2$, with equality iff $u = \alpha v$.
- **Ridge Regression Gradient:** $J(w) = \frac{1}{2n}\sum_{i=1}^n (w^T x_i - y_i)^2 + \frac{\lambda}{2}\|w\|_2^2$ has gradient $\nabla_w J(w) = \frac{1}{n}\sum_{i=1}^n (w^T x_i - y_i)x_i + \lambda w$, where $L_2$ regularization acts as weight decay $(1 - \eta\lambda)w^{(t)}$.

### 1.2 Probability Foundations & Maximum Likelihood
- **Bernoulli MLE:** $\hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n X_i$.
- **Univariate Gaussian MLE:** $\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n X_i$ and $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \hat{\mu})^2$.

---

## Section 2: Sentiment Classification (`sentiment`)
**Author:** SSEWAJE JOLLY HILTON

### 2.1 Problem 1: Building Intuition

**Part a: SGD by Hand**
We run stochastic gradient descent (SGD) on the hinge loss. Let the feature mapping \(\phi(x)\) return word counts and let the initial weight vector be \(\mathbf{w} = [0, 0, 0, 0, 0, 0]\) for the tokens ("pretty", "good", "bad", "plot", "not", "scenery"). We use a learning rate \(\eta = 0.1\).

1. **"not good" (y = -1):** Score = 0. Margin = 0 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(-1)\phi\). 
   \(\mathbf{w} = [0, -0.1, 0, 0, -0.1, 0]\)
2. **"pretty bad" (y = -1):** Score = 0. Margin = 0 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(-1)\phi\).
   \(\mathbf{w} = [-0.1, -0.1, -0.1, 0, -0.1, 0]\)
3. **"good plot" (y = +1):** Score = -0.1. Margin = -0.1 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(+1)\phi\).
   \(\mathbf{w} = [-0.1, 0.0, -0.1, 0.1, -0.1, 0]\)
4. **"pretty scenery" (y = +1):** Score = -0.1. Margin = -0.1 < 1. Update: \(\mathbf{w} \leftarrow \mathbf{w} + 0.1(+1)\phi\).
   \(\mathbf{w} = [0.0, 0.0, -0.1, 0.1, -0.1, 0.1]\)

**Final Weight Vector:** `[0.0, 0.0, -0.1, 0.1, -0.1, 0.1]`

**Part b: Linear Separability Proof**
Let the weights for "bad", "good", and "not" be \(w_b, w_g, w_n\). For zero error, the classifier must satisfy:
1. "bad" (-1): \(w_b < 0\)
2. "good" (+1): \(w_g > 0\)
3. "not bad" (+1): \(w_n + w_b > 0 \implies w_n > -w_b > 0\)
4. "not good" (-1): \(w_n + w_g < 0 \implies w_n < -w_g < 0\)

We cannot have \(w_n > 0\) and \(w_n < 0\) simultaneously. Thus, no linear classifier can perfectly separate this dataset.

**Additional Feature:** Add a bigram feature for **"not_bad"** and **"not_good"**. This allows the model to learn a positive weight for "not_bad" and a negative weight for "not_good" independent of the individual word weights.

### 2.2 Problem 2: Predicting Movie Ratings

**Part a: Squared Loss**
\[
 \text{Loss}(x, y, \mathbf{w}) = \frac{1}{2} (\sigma(\mathbf{w} \cdot \phi(x)) - y)^2 
\]

**Part b: Gradient**
Let \(p = \sigma(\mathbf{w} \cdot \phi(x))\). The derivative of the sigmoid is \(\sigma'(z) = p(1 - p)\).
\[
 \nabla_\mathbf{w} \text{Loss} = (p - y) \cdot p(1 - p) \cdot \phi(x) 
\]

**Part c: Magnitude of Gradient**
To make the gradient arbitrarily small, we need \(p \to 1\) (since \(y=1\)). This happens when \(\mathbf{w} \cdot \phi(x) \to \infty\). Thus, making the weights arbitrarily large in the direction of \(\phi(x)\) will shrink the gradient. The gradient can **never be exactly zero** for finite weights because the sigmoid function only approaches 1 asymptotically. Since we cannot set weights to infinity, \(1-p\) will always be strictly greater than zero.

### 2.3 Problem 3: Sentiment Classification

**Part e: Character N-grams**
For the value of \(n\), \(n=3\) or \(n=4\) usually produces the smallest validation error. Character n-grams perform almost as well as word features because they capture sub-word morphological patterns (like prefixes and suffixes) and handle misspellings or punctuation.
A review where character n-grams outperform word features: *"I hated this movie. It was un-freaking-believable."* Word features would see "un-freaking-believable" as one unknown word, but character n-grams (like "fre", "eak", "bel") can match it to the positive root word "believable", capturing the sentiment better.

### 2.4 Problem 5: K-means Clustering

**Part a: Manual 2-means Execution**
*Run 1:* Initial \(\mu_1=[0,0], \mu_2=[11,0]\). Final centers: \(\mu_1=[2,0], \mu_2=[8.5,0]\). Assignments: \(x_1, x_2\) in cluster 1; \(x_3, x_4\) in cluster 2. Final loss: 20.5.
*Run 2:* Initial \(\mu_1=[0,0], \mu_2=[4,0]\). Final centers: \(\mu_1=[0,0], \mu_2=[7,0]\). Assignments: \(x_1\) in cluster 1; \(x_2, x_3, x_4\) in cluster 2. Final loss: 26.

**Part c: Scaling K-means**
*Scaling all dimensions by a non-zero factor:* **Yes**, the clusters remain the same. K-means uses Euclidean distance, and scaling all dimensions by \(c\) multiplies all distances by \(c^2\). Since the relative ordering of distances doesn't change, the nearest centroid remains identical.
*Scaling only certain dimensions:* **No**, the clusters are not guaranteed to remain the same. Scaling one dimension changes the relative distances between points. For example, points A=[1,1], B=[2,1], C=[1,10] might cluster differently if we multiply only the first dimension by 10.


## Section 3: Route Planning (`route`)
**Author:** Atwiine Precious Mwebaze (Student No. 2500703347)

### 3.1 Admissibility and Consistency of Heuristics
- **Definitions & Mathematical Proof:** $h(s)$ is admissible if $0 \le h(s) \le h^*(s)$ and consistent if $h(s) \le c(s, a, s') + h(s')$ with $h(g) = 0$. Telescoping along an optimal path $s = s_0, \dots, s_k = g$ proves $h(s_0) \le h^*(s_0) + h(g) = h^*(s)$, so consistency implies admissibility.
- **Landmark Heuristic Construction:** $h_L(s) = |d(s, L) - d(g, L)|$ satisfies $h_L(u) - h_L(v) \le |d(u, L) - d(v, L)| \le c(u, v)$ by the triangle and reverse triangle inequalities, with $h_L(g) = 0$, making it consistent and admissible.

### 3.2 Algorithm Comparison: Dijkstra vs. A*
- **Search Space Expansion:** UCS/Dijkstra expands in concentric contours of $g(s)$, while A* directs search along an ellipse toward the goal using $f(s) = g(s) + h(s)$.
- **Optimality Trade-offs:** With a consistent heuristic, A* guarantees optimal paths without re-opening closed nodes while expanding a subset of UCS states.

---

## Section 4: Controlling MountainCar (`mountaincar`)
**Authors:** Opolot Emmanuel Marcus (Student No. 2500729501) & Kabanda Joel (Student No. 2500703381)

### 4.1 Markov Decision Process (MDP) Formulation
- **State Space, Action Space, Transition Dynamics:** $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$ with $s = (x, v) \in [-1.2, 0.6] \times [-0.07, 0.07]$, $a \in \{-1, 0, +1\}$, $R = -1$ for $x < 0.5$ and $0$ at goal $x \ge 0.5$.
- **Bellman Optimality Equations:** $(T^* V)(s) = \max_a [R(s, a) + \gamma \sum_{s'} T(s, a, s') V(s')]$.

### 4.2 Value Iteration vs. Policy Iteration vs. Q-Learning
- **Contraction Mapping Proof Outline:** Using $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$, $\|T^* U - T^* V\|_\infty \le \gamma \|U - V\|_\infty$, so by the Banach Fixed-Point Theorem Value Iteration converges linearly at rate $\gamma$ to $V^*$.
- **Convergence Properties under Function Approximation:** Semi-gradient Q-learning $\hat{Q}(s, a; w) = w \cdot \phi(s, a)$ updates $w \leftarrow w + \alpha (r + \gamma \max_{a'} \hat{Q}(s', a'; w) - \hat{Q}(s, a; w))\phi(s, a)$; coarse tile coding and small $\alpha$ mitigate the deadly triad.

---

## Section 5: Pac-Man Multi-Agent Adversarial Search (`pacman`)
**Authors:** Mukisa Percy (Student No. 2500703468) & Munanura Jill (Student No. 2500705824)

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
**Author:** Joseph Mary Yiga (Student No. 2500703615)

### 6.1 Constraint Satisfaction Problems (CSP)
- **Constraint Graph Formulation:**
  A CSP is a tuple $(X, D, C)$ where $X = \{X_1, \dots, X_n\}$ are variables (courses), $D = \{D_1, \dots, D_n\}$ are finite domains (quarters), and $C$ are factors $f_j(x) \ge 0$ enforcing prerequisites ($q_{\text{prereq}} < q_{\text{course}}$) and scheduling rules.
- **Arc Consistency (AC-3) & Forward Checking Complexity:**
  With $e$ binary constraints and maximum domain size $d$, each directed arc $(X_i, X_j)$ can be re-enqueued at most $d$ times, and checking an arc takes $O(d^2)$ pairs. Thus AC-3 has worst-case time complexity **$O(e \cdot d^3)$**. Combined with **MRV** (fail-first) and **LCV** (fail-last), AC-3 drastically reduces backtracking.

### 6.2 Bayesian Networks & Conditional Independence
- **D-separation Queries:**
  A path between $A$ and $B$ given evidence $E$ is blocked if it contains a triple $(X, Y, Z)$ that is a **chain** ($X \to Y \to Z$) or **fork** ($X \leftarrow Y \to Z$) with $Y \in E$, or a **collider** ($X \to Y \leftarrow Z$) with neither $Y$ nor any descendant of $Y$ in $E$ (observing a collider activates *explaining away*).
- **Exact Inference (Variable Elimination) vs. Approximate Sampling:**
  Exact inference computes $P(Q \mid E)$ exactly in $O(n \cdot d^{w+1})$ (exponential in induced treewidth $w$), whereas **Likelihood Weighting** fixes $E$, samples non-evidence nodes in topological order, and weights samples by $\prod_{e \in E} P(e \mid \text{Parents}(e))$, converging at $O(1/\sqrt{N})$.

---

## Section 7: Car Tracking & From Language to Logic (`tracking` / `logic`)
**Author:** Muzira Joshua Kwizera (Student No. 2500703484)

### 7.1 Hidden Markov Models (HMM) & Particle Filtering
- **Forward-Backward Algorithm Derivation:**
  Filtering computes $P(H_t \mid e_{1:t}) \propto P(e_t \mid H_t) \sum_{h_{t-1}} P(H_t \mid h_{t-1}) P(h_{t-1} \mid e_{1:t-1})$ in $O(T |\mathcal{S}|^2)$ time over $T$ steps and $|\mathcal{S}|$ states.
- **Particle Depletion & Importance Resampling:**
  Particle filtering approximates $P(H_t \mid e_{1:t})$ with $N$ particles: propose via transition model, weight by Gaussian sensor likelihood $P(e_t \mid h_t^{(i)})$, and apply **systematic resampling** to prevent particle degeneracy/depletion.

### 7.2 Propositional & First-Order Logic
- **Conjunctive Normal Form (CNF) Transformation:**
  Eliminate $\leftrightarrow$ and $\to$, push $\neg$ inward via De Morgan's laws and double-negation elimination, and distribute $\lor$ over $\land$ to obtain a set of disjunctions of literals.
- **Resolution Proofs and Soundness/Completeness:**
  From $(A \lor p)$ and $(B \lor \neg p)$, Resolution derives $(A \lor B)$, which is **sound** (every model of the premises satisfies the resolvent) and **refutation-complete** ($\text{KB} \models \alpha \iff \text{CNF}(\text{KB} \land \neg\alpha) \vdash_{\text{Res}} \square$).
