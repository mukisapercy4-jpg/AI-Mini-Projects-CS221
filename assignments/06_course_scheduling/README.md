# Assignment 6: Course Scheduling (CSP) & Bayesian Networks

**Assigned Member:** Joseph Mary Yiga (Student No. 2500703615)  
**Reference:** [Stanford CS221 Coursework (`scheduling` / `bayesian`)](https://stanford-cs221.github.io/spring2026/#coursework)

---

## 🎯 Objectives & Tasks

### 1. Programming Tasks
1. **Constraint Satisfaction Problem (`CSP`)**:
   - Represent variables, finite domains, unary factors, and binary factors.
2. **Backtracking Search (`BacktrackingSearch`)**:
   - Standard backtracking, **MCV / MRV** (Most Constrained Variable), **LCV** (Least Constraining Value), and **AC-3** (Arc Consistency) lookahead.
3. **Course Scheduling Formulation (`create_course_scheduling_csp`)**:
   - Enforce prerequisite ordering, quarter/semester capacity limits, and no-overlap constraints.
4. **Bayesian Networks (`BayesianNetwork`)**:
   - Exact inference via enumeration / variable elimination (`exact_inference`).
   - Approximate inference via likelihood weighting (`likelihood_weighting`).

### 2. Theoretical Tasks
1. Constraint graph formulation and AC-3 worst-case time complexity $O(e d^3)$.
2. D-separation rules in Bayesian networks (chain, fork, collider / explaining away).
3. Exact inference (Variable Elimination) vs. Approximate Sampling (Gibbs / Likelihood Weighting).

---

## 📁 Module File Structure
- `submission_scheduling.py`: CSP solver (Backtracking, MRV, LCV, AC-3) and Bayesian Network inference.
- `scheduling_engine.py`: University course catalog, prerequisites, and sample Bayesian alarm/medical network.
- `test_scheduling.py`: Automated unit test suite.
- `run_scheduling_experiments.py`: Benchmark comparing plain backtracking vs. MRV+AC-3, and exact vs. approximate Bayesian inference.
- `theoretical_scheduling.md`: Written theoretical solutions.
- `HOW_TO_TEST.md`: Quick verification guide.
