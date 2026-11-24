# Theoretical Analysis: Assignment 4 — Controlling MountainCar

**Authors:** Marcus & Co-Lead  
**Course:** Makerere University — AI Mini-Projects (CS221 Aligned)

---

## 1. Markov Decision Process (MDP) Formulation

An MDP is defined by the 5-tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$:
- **State Space $\mathcal{S}$:** Continuous 2D state $s = (x, v)$ where position $x \in [-1.2, 0.6]$ and velocity $v \in [-0.07, 0.07]$.
- **Action Space $\mathcal{A}$:** Discrete throttle commands $a \in \{-1 \text{ (Reverse)}, 0 \text{ (Coast)}, +1 \text{ (Forward)}\}$.
- **Transition Dynamics $T(s, a, s')$:** Deterministic non-linear physics dictated by Newton's second law and sinusoidal gravitational force:
  $$v_{t+1} = \text{clip}(v_t + 0.001 a_t - 0.0025 \cos(3 x_t), -0.07, 0.07)$$
  $$x_{t+1} = \text{clip}(x_t + v_{t+1}, -1.2, 0.6)$$
- **Reward Function $R(s, a, s')$:** $R = -1.0$ for non-goal states ($x < 0.5$) and $0.0$ at terminal states ($x \ge 0.5$).
- **Discount Factor:** $\gamma \in (0, 1)$, typically $0.99$.

---

## 2. Contraction Mapping Proof for Bellman Optimality

The Bellman optimality operator $T^*$ maps a value function $V \in \mathbb{R}^{|\mathcal{S}|}$ to another value function:
$$(T^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} T(s, a, s') V(s') \right]$$

### Theorem:
$T^*$ is a contraction mapping in the supremum norm with contraction factor $\gamma < 1$:
$$\|T^* U - T^* V\|_\infty \le \gamma \|U - V\|_\infty \quad \forall U, V$$

### Proof:
For any state $s$:
$$\begin{aligned}
(T^* U)(s) - (T^* V)(s) &= \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s'} T(s, a, s') U(s') \right] - \max_{a' \in \mathcal{A}} \left[ R(s, a') + \gamma \sum_{s'} T(s, a', s') V(s') \right]
\end{aligned}$$
Using the algebraic inequality $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$:
$$\begin{aligned}
|(T^* U)(s) - (T^* V)(s)| &\le \max_{a \in \mathcal{A}} \left| \left( R(s, a) + \gamma \sum_{s'} T(s, a, s') U(s') \right) - \left( R(s, a) + \gamma \sum_{s'} T(s, a, s') V(s') \right) \right| \\
&= \max_{a \in \mathcal{A}} \gamma \left| \sum_{s'} T(s, a, s') (U(s') - V(s')) \right| \\
&\le \max_{a \in \mathcal{A}} \gamma \sum_{s'} T(s, a, s') |U(s') - V(s')| \\
&\le \max_{a \in \mathcal{A}} \gamma \sum_{s'} T(s, a, s') \|U - V\|_\infty \\
&= \gamma \|U - V\|_\infty \quad \left(\text{since } \sum_{s'} T(s, a, s') = 1\right)
\end{aligned}$$
Taking the supremum over all states $s \in \mathcal{S}$:
$$\|T^* U - T^* V\|_\infty \le \gamma \|U - V\|_\infty$$
By the **Banach Fixed-Point Theorem**, since $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$ is a complete metric space, Value Iteration $V_{k+1} = T^* V_k$ converges linearly at rate $\gamma$ to the unique fixed point $V^* = T^* V^*$. $\blacksquare$

---

## 3. Reinforcement Learning with Function Approximation

### 3.1 Semi-Gradient Q-Learning
When continuous state spaces prohibit tabular representation, we approximate $Q(s, a)$ with parameter vector $w$:
$$\hat{Q}(s, a; w) = w \cdot \phi(s, a)$$
The Bellman error objective is:
$$J(w) = \frac{1}{2} \mathbb{E} \left[ \left( r + \gamma \max_{a'} \hat{Q}(s', a'; w) - \hat{Q}(s, a; w) \right)^2 \right]$$
Taking the semi-gradient (treating the target $r + \gamma \max_{a'} \hat{Q}$ as fixed without propagating gradients through the max operator):
$$\nabla_w J(w) \approx -\left( r + \gamma \max_{a'} \hat{Q}(s', a'; w) - \hat{Q}(s, a; w) \right) \phi(s, a)$$
giving the stochastic update:
$$w \leftarrow w + \alpha \cdot \text{TD\_error} \cdot \phi(s, a)$$

### 3.2 The Deadly Triad
Stability of TD algorithms is threatened when three elements coincide:
1. **Function Approximation:** Using linear combinations or neural networks rather than tabular lookups.
2. **Bootstrapping:** Updating estimates based on subsequent estimates ($r + \gamma \max Q$) rather than Monte Carlo empirical returns.
3. **Off-Policy Training:** Learning the optimal policy $\pi^*$ while following an exploratory policy $\mu$ ($\epsilon$-greedy).
In our implementation, coarse tile coding and conservative step sizes ($\alpha \le 0.1$) ensure stability and rapid convergence on MountainCar.
