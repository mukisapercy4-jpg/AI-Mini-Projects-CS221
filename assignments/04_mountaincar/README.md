# Assignment 4: Controlling MountainCar (MDPs & Reinforcement Learning)

**Assigned Members:** Marcus & Co-Lead  
**Reference:** [Stanford CS221 MountainCar Coursework](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

In this assignment, you solve the classic MountainCar control problem where an underpowered car must drive up a steep hill by first building momentum in reverse:

### 1. Programming Tasks
1. **MountainCar Dynamics Engine (`mountaincar_engine.py`)**:
   - Continuous state space: Position $x \in [-1.2, 0.6]$, Velocity $v \in [-0.07, 0.07]$.
   - Discrete actions: $a \in \{-1 \text{ (Left)}, 0 \text{ (Coast)}, +1 \text{ (Right)}\}$.
   - Non-linear physics:
     $$v_{t+1} = \text{clip}(v_t + 0.001 a_t - 0.0025 \cos(3 x_t))$$
     $$x_{t+1} = \text{clip}(x_t + v_{t+1})$$
   - Reward: $-1.0$ per time step until goal ($x \ge 0.5$) is reached.
2. **Markov Decision Process & Value Iteration (`ValueIteration`)**:
   - State discretization into fine grid bins.
   - Bellman Optimality update:
     $$V_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V_k(s')]$$
3. **Model-Free Reinforcement Learning (Q-Learning with Function Approximation)**:
   - Tile coding / radial basis features $\phi(s, a)$.
   - Approximate action-value function:
     $$\hat{Q}(s, a; w) = w \cdot \phi(s, a)$$
   - Semi-gradient TD update:
     $$w \leftarrow w + \alpha [r + \gamma \max_{a'} \hat{Q}(s', a'; w) - \hat{Q}(s, a; w)] \phi(s, a)$$
   - $\epsilon$-greedy exploration with decay.

### 2. Theoretical Tasks
1. **Bellman Optimality & Contraction Mapping Theorem**:
   - Mathematical proof that the Bellman optimality operator $T^*$ is a $\gamma$-contraction in supremum norm ($\|T^* U - T^* V\|_\infty \le \gamma \|U - V\|_\infty$).
   - Banach Fixed-Point Theorem guarantee of unique optimal value function $V^*$.
2. **Convergence Guarantees & The Deadly Triad**:
   - Conditions under which tabular Q-learning converges almost surely to $Q^*$.
   - Why combining function approximation, bootstrapping, and off-policy training (the Deadly Triad) can lead to divergence.

---

## 📁 Module File Structure

- `submission_mountaincar.py`: MDP models, Discretized Value Iteration, Q-Learning with Function Approximation.
- `mountaincar_engine.py`: Continuous 1D MountainCar simulator environment.
- `test_mountaincar.py`: Automated unit tests.
- `run_mountaincar_experiments.py`: Benchmark runner training RL agent and tracking episode rewards.
- `theoretical_mountaincar.md`: Formal contraction proofs and reinforcement learning writeup.
- `HOW_TO_TEST.md`: Instructions for Marcus and reviewers.
