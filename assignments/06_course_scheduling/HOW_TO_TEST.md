# How to Test — Assignment 6: Course Scheduling & Bayesian Networks

**Author:** Joseph Mary Yiga (Student No. 2500703615)  
**Module:** `assignments/06_course_scheduling/`

---

## Quick Start

```powershell
python assignments/06_course_scheduling/test_scheduling.py
```

Expected output:
```text
Testing BacktrackingSearch on 3-Coloring CSP...
  [PASS] 3-Coloring solved...
Testing Course Scheduling CSP with prerequisites...
  [PASS] Course Schedule valid...
Testing AC-3 unsatisfiability detection...
  [PASS] Unsatisfiable K3-with-2-colors detected cleanly.
Testing BayesianNetwork exact inference (Alarm Network)...
  [PASS] Exact P(B=True | J=T, M=T) = 0.28417
Testing BayesianNetwork likelihood weighting...
  [PASS] Likelihood Weighting P(B=True | J=T, M=T) = ...

ALL COURSE SCHEDULING & BAYESIAN NETWORK TESTS PASSED!
```

## Run Benchmark Experiments

```powershell
python assignments/06_course_scheduling/run_scheduling_experiments.py
```
