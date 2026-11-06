# How to Test the Pac-Man Module

**Assigned Reviewers & Testers:** Jill, Mukisa Percy, and the Integration Lead  
**Branch:** `mukisa`

---

## 🧪 1. Quick Automated Unit Tests (30 seconds)
To verify that all classes, minimax trees, pruning checks, and evaluation functions operate without syntax or logic errors:

```powershell
# Inside project root
python assignments/05_pacman/test_pacman.py
```

### Expected Output:
```text
Testing Manhattan Distance...
  [PASS] Manhattan distance calculation.
Testing ReflexAgent...
  [PASS] ReflexAgent initialized.
Testing MinimaxAgent...
  [PASS] MinimaxAgent selected action: North
Testing AlphaBetaAgent...
  [PASS] AlphaBetaAgent selected action: North
Testing ExpectimaxAgent...
  [PASS] ExpectimaxAgent selected action: North
Testing betterEvaluationFunction...
  [PASS] betterEvaluationFunction returned value: 7.27
ALL BASIC PAC-MAN TESTS PASSED!
```

---

## 📊 2. Full Multi-Agent Benchmark Simulation (2-3 minutes)
To benchmark all 4 agents (Reflex, Minimax, AlphaBeta, and Expectimax) across randomized seeds and verify pruning efficiency:

```powershell
python assignments/05_pacman/run_pacman_experiments.py
```

### What Jill / Percy Should Verify:
1. **Minimax vs. AlphaBeta Decision Equivalence:**
   - On the same trial seeds, `MinimaxAgent` and `AlphaBetaAgent` should select identical game actions and achieve identical game scores.
2. **Pruning Speedup:**
   - `AlphaBetaAgent` move latency should be lower than unpruned `MinimaxAgent` (e.g., ~9 ms vs ~14 ms per move).
3. **Score & Survival:**
   - Both `ReflexAgent` and `ExpectimaxAgent` should achieve high scores (e.g., 800+ on wins) due to food-seeking and ghost-avoidance heuristic weights.

---

## 📓 3. Master Notebook Verification
To verify that the master submission notebook runs end-to-end:

1. Launch Jupyter Notebook or VSCode:
   ```powershell
   jupyter notebook notebooks/ai_coursework_all_assignments.ipynb
   ```
2. Click **Kernel -> Restart Kernel and Run All Cells**.
3. Scroll to **Section 5 (Assignment 5: Pac-Man)**:
   - Check that the grid engine and agent classes load cleanly.
   - Verify that the benchmark table prints properly.
   - Verify that the comparative **Matplotlib performance bar chart** is rendered.

---

## 📄 4. Theoretical Document Verification
- Open `theoretical/theoretical_part_master.md` or compile `theoretical/theoretical_part_master.tex`.
- Verify Section 5 contains:
  - Multi-agent game value recurrence equations.
  - Sub-optimal ghost behavior explanation.
  - Multi-agent $\alpha$-cutoff and $\beta$-cutoff conditions with $O(b^{d/2})$ complexity proof.
  - Monotonic vs. positive affine transformation invariance proof (Minimax vs. Expectimax).
  - Mathematical formulation of `betterEvaluationFunction`.
