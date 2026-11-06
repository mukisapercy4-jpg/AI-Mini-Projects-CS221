# Assignment 5: Pac-Man (Multi-Agent Adversarial Search)

**Assigned Members:** Mukisa Percy & Jill  
**Branch:** `mukisa`  
**Reference:** [Stanford CS221 Pac-Man Coursework](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

In this assignment, you design agents for the classic game of Pac-Man, both reflex agents and multi-agent adversarial search algorithms:

### 1. Programming Tasks
1. **Reflex Agent (`ReflexAgent`)**:
   - Design an evaluation function evaluating state-action pairs.
   - Balance reward from eating food dots and penalty from proximity to active ghosts.
2. **Minimax Agent (`MinimaxAgent`)**:
   - Implement multi-agent Minimax search.
   - Pac-Man is the MAX agent (agent index `0`).
   - Multiple ghosts are MIN agents (agent indices `1, 2, ...`).
   - One search "ply" consists of 1 Pac-Man move followed by every ghost taking a move.
3. **Alpha-Beta Pruning Agent (`AlphaBetaAgent`)**:
   - Optimize the Minimax tree search using Alpha-Beta pruning.
   - Pruning occurs across multiple ghosts per ply ($\alpha$ and $\beta$ propagated through the multi-agent layers).
4. **Expectimax Agent (`ExpectimaxAgent`)**:
   - Model scenarios where ghosts behave sub-optimally or probabilistically (e.g. random moves).
   - Pac-Man maximizes, while ghosts take expectations (averages) over legal moves.
5. **Better Evaluation Function (`betterEvaluationFunction`)**:
   - Evaluate states (rather than state-action pairs).
   - Features to incorporate:
     - Distance to closest food pellet
     - Number of remaining food pellets
     - Distance to active ghosts vs. scared ghosts (after eating a power capsule)
     - Distance to capsules
     - Game score

### 2. Theoretical Tasks
1. **Minimax Properties**:
   - Prove conditions under which Minimax guarantees optimal gameplay against optimal adversaries.
   - What happens if the adversary does not play optimally?
2. **Alpha-Beta Pruning Theorems**:
   - Derive the best-case branching factor and leaf evaluations with optimal move ordering ($O(b^{d/2})$).
   - Describe exact conditions where a branch can be safely pruned.
3. **Expectimax vs Minimax**:
   - Explain why positive linear transformations of the evaluation function affect Minimax and Expectimax differently (scale invariance vs. scale sensitivity).

---

## 🚀 How Percy & Jill Should Collaborate on this Branch

1. You and Jill can commit and push directly to this branch:
   ```bash
   git checkout mukisa
   git pull origin mukisa
   ```
2. When testing code, you can run experiments inside `assignments/05_pacman/`.
3. Once validated, copy your solutions into Section 5 of `notebooks/ai_coursework_all_assignments.ipynb` and `theoretical/theoretical_part_master.tex` / `.md`.
