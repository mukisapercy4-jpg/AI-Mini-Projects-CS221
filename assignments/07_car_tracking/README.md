# Assignment 7: Car Tracking (HMM & Particle Filtering) & Logic

**Assigned Member:** Muzira Joshua Kwizera (Student No. 2500703484)  
**Reference:** [Stanford CS221 Coursework (`car` / `logic`)](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

### 1. Programming Tasks
1. **Exact HMM Inference (`ExactInference`)**:
   - Maintain discrete belief distribution $B(r, c)$ over a 2D grid.
   - `observe(agent_x, agent_y, observed_dist)`: Bayes update with Gaussian sensor likelihood.
   - `elapse_time()`: Forward transition update across grid cells.
2. **Particle Filtering (`ParticleFilter`)**:
   - Approximate belief with $N$ weighted particles.
   - Importance weighting + systematic resampling to avoid particle depletion.
3. **Propositional Logic & Resolution (`to_cnf`, `resolution_entails`)**:
   - AST formula classes (`Atom`, `Not`, `And`, `Or`, `Implies`, `Equiv`).
   - Conversion to Conjunctive Normal Form (CNF) and Resolution refutation theorem prover.

### 2. Theoretical Tasks
1. Forward-Backward HMM recurrence and complexity $O(T |\mathcal{S}|^2)$.
2. Particle depletion and importance resampling.
3. CNF transformation steps (implication elimination, De Morgan's laws, distribution) and soundness/completeness of Resolution.

---

## 📁 Module File Structure
- `submission_car_tracking.py`: `ExactInference`, `ParticleFilter`, and Propositional Logic / CNF / Resolution engine.
- `car_tracking_engine.py`: 2D city grid car simulator with noisy sonar/GPS distance readings.
- `test_car_tracking.py`: Automated unit test suite.
- `run_car_tracking_experiments.py`: Benchmark comparing Exact HMM vs. Particle Filter localization error + Logic theorem proving demo.
- `theoretical_car_tracking.md`: Written theoretical derivations.
- `HOW_TO_TEST.md`: Quick verification guide.
