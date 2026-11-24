# Assignment 3: Route Planning (State-Space Search & A*)

**Assigned Member:** Precious Atwiine  
**Reference:** [Stanford CS221 Route Planning Coursework](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

In this assignment, you implement state-space search algorithms for GPS route planning on city road networks:

### 1. Programming Tasks
1. **Search Problem Interface (`SearchProblem`)**:
   - `start_state()`: Returns initial state.
   - `is_end(state)`: Goal test.
   - `successors_and_costs(state)`: Yields `(action, next_state, cost)`.
2. **Uniform Cost Search (`uniform_cost_search`)**:
   - Priority queue ordered strictly by accumulated path cost $g(s)$.
   - Guarantees optimal (shortest) path on non-negative edge weights.
3. **A* Search (`a_star_search`)**:
   - Priority queue ordered by estimated total cost $f(s) = g(s) + h(s)$.
   - Dynamic exploration tracking to prune dominated paths.
4. **Heuristic Functions**:
   - **Euclidean / Manhattan Distance Heuristic**: Straight-line physical lower bounds.
   - **Landmark Heuristic (`LandmarkHeuristic`)**: Precomputed shortest paths to selected landmark nodes utilizing the triangle inequality:
     $$h_L(s) = |d(s, L) - d(\text{goal}, L)|$$

### 2. Theoretical Tasks
1. **Admissibility & Consistency**:
   - Mathematical definitions:
     - Admissibility: $\forall s, h(s) \le h^*(s)$.
     - Consistency: $\forall s, a, s', h(s) \le \text{cost}(s, a, s') + h(s')$.
   - Proof that consistency strictly implies admissibility when $h(\text{goal}) = 0$.
2. **Landmark Heuristic Consistency Proof**:
   - Proof using the triangle inequality that $h_L(s) = |d(s, L) - d(g, L)|$ is consistent and admissible.
3. **Search Space Reduction**:
   - Comparative analysis of node expansions between UCS and A* under consistent heuristics.

---

## 📁 Module File Structure

- `submission_route.py`: Core search algorithms (UCS, A*, Landmark Heuristic).
- `route_engine.py`: City grid road network and graph simulation.
- `test_route.py`: Automated unit tests.
- `run_route_experiments.py`: Benchmark runner comparing UCS vs. A* node expansions and latency.
- `theoretical_route.md`: Complete theoretical proofs and optimality guarantees.
- `HOW_TO_TEST.md`: Instructions for Precious and team reviewers.
