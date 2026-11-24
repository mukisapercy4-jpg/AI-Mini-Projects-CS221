# How to Test Assignment 3: Route Planning

**Assigned Reviewer:** Precious Atwiine & Integration Lead  
**Folder:** `assignments/03_route_planning/`

---

## 🧪 1. Quick Automated Unit Tests
Run the standalone unit tests:
```powershell
python assignments/03_route_planning/test_route.py
```
Expected output:
```text
Testing City Map Network Graph...
  [PASS] City map graph initialized.
Testing Uniform Cost Search (UCS)...
  [PASS] UCS found path: Downtown -> TechPark -> Airport with cost: 8.50 (Expanded: ...)
Testing A* Search with Euclidean Heuristic...
  [PASS] A* (Euclidean) found optimal path with cost: 8.50 (Expanded: ...)
Testing Landmark Heuristic Admissibility & Consistency...
  [PASS] A* (Landmark) found optimal path with cost: 8.50 (Expanded: ...)
  [PASS] Landmark heuristic proven consistent on all edges.
ALL BASIC ROUTE PLANNING TESTS PASSED!
```

---

## 📊 2. Node Expansion Benchmarks (UCS vs. A*)
Run comparative benchmarking across alternative routes:
```powershell
python assignments/03_route_planning/run_route_experiments.py
```

---

## 📄 3. Theoretical Verification
- Review `theoretical_route.md`.
- Verify the mathematical proofs for:
  - Consistency implies admissibility ($h(s) \le h^*(s)$).
  - Landmark heuristic $h_L(s) = |d(s, L) - d(g, L)|$ consistency using triangle inequality.
  - Search wavefront comparison between UCS and A*.
