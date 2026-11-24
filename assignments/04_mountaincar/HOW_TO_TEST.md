# How to Test Assignment 4: Controlling MountainCar

**Assigned Reviewers:** Marcus, Co-Lead & Integration Lead  
**Folder:** `assignments/04_mountaincar/`

---

## 🧪 1. Quick Automated Unit Tests
Run the standalone unit tests:
```powershell
python assignments/04_mountaincar/test_mountaincar.py
```
Expected output:
```text
Testing MountainCar Physical Simulator...
  [PASS] Physics simulator step verified.
Testing Discretized MountainCar MDP & Value Iteration...
  [PASS] Value iteration converged on discretized grid.
Testing Feature Extractor & Q-Learning Agent...
  [PASS] Q-learning agent update verified.
ALL BASIC MOUNTAINCAR TESTS PASSED!
```

---

## 📊 2. Value Iteration & RL Benchmark Training
Run the training experiments:
```powershell
python assignments/04_mountaincar/run_mountaincar_experiments.py
```

---

## 📄 3. Theoretical Verification
- Review `theoretical_mountaincar.md`.
- Verify the mathematical derivations for:
  - 5-tuple continuous MDP definition.
  - $\gamma$-contraction proof for Bellman optimality operator $T^*$ in $\ell_\infty$ norm.
  - Semi-gradient Q-learning derivation and the Deadly Triad conditions.
