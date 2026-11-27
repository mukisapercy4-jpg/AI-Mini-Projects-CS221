"""
Unit Test Suite for Assignment 6: Course Scheduling (CSP) & Bayesian Networks
Author: Joseph Mary Yiga (Student No. 2500703615)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from scheduling_engine import get_sample_course_catalog, get_alarm_bayesian_network
from submission_scheduling import (
    CSP,
    BacktrackingSearch,
    create_course_scheduling_csp,
    BayesianNetwork,
)


def run_tests():
    # 1. Basic Map Coloring CSP -----------------------------------------------
    print("Testing BacktrackingSearch on 3-Coloring CSP...")
    csp = CSP()
    for region in ["WA", "NT", "SA", "Q"]:
        csp.add_variable(region, ["R", "G", "B"])
    edges = [("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"), ("SA", "Q")]
    for u, v in edges:
        csp.add_binary_factor(u, v, lambda c1, c2: 1.0 if c1 != c2 else 0.0)

    solver = BacktrackingSearch(csp, mrv=True, lcv=True, ac3=True)
    sol = solver.solve()
    assert sol is not None, "Expected a valid 3-coloring"
    for u, v in edges:
        assert sol[u] != sol[v], f"Adjacent regions {u} and {v} have same color"
    print(f"  [PASS] 3-Coloring solved in {solver.num_assignments} assignments: {sol}")

    # 2. Course Scheduling CSP -------------------------------------------------
    print("Testing Course Scheduling CSP with prerequisites...")
    catalog = get_sample_course_catalog()
    sched_csp = create_course_scheduling_csp(catalog)
    sched_solver = BacktrackingSearch(sched_csp, mrv=True, lcv=True, ac3=True)
    schedule = sched_solver.solve()
    assert schedule is not None, "Expected a valid course schedule"
    for prereq, course in catalog["prerequisites"]:
        assert schedule[prereq] < schedule[course], (
            f"Prerequisite {prereq} (Q{schedule[prereq]}) must precede {course} (Q{schedule[course]})"
        )
    print(f"  [PASS] Course Schedule valid: {schedule}")

    # 3. Unsatisfiable CSP Detection via AC-3 ----------------------------------
    print("Testing AC-3 unsatisfiability detection...")
    unsat = CSP()
    for x in ["A", "B", "C"]:
        unsat.add_variable(x, [1, 2])
    for u, v in [("A", "B"), ("B", "C"), ("A", "C")]:
        unsat.add_binary_factor(u, v, lambda x, y: 1.0 if x != y else 0.0)
    unsat_solver = BacktrackingSearch(unsat, mrv=True, lcv=True, ac3=True)
    assert unsat_solver.solve() is None, "K3 with 2 colors must be unsatisfiable"
    # K3 with 2 colors is initially 2-arc-consistent, so AC-3 detects failure after 1 assignment
    assert unsat_solver.num_assignments <= 2, f"Expected early AC-3 failure, got {unsat_solver.num_assignments}"
    print(f"  [PASS] Unsatisfiable K3-with-2-colors detected in {unsat_solver.num_assignments} assignment(s).")

    # 4. Bayesian Network Exact Inference --------------------------------------
    print("Testing BayesianNetwork exact inference (Alarm Network)...")
    net_data = get_alarm_bayesian_network()
    bn = BayesianNetwork(net_data["nodes"], net_data["parents"], net_data["cpts"])
    post = bn.exact_inference("B", {"J": True, "M": True})
    # Classic AIMA / Pearl result: P(Burglary | JohnCalls=T, MaryCalls=T) ≈ 0.28417
    assert abs(post[True] - 0.28417) < 1e-3, f"Expected ~0.28417, got {post[True]:.5f}"
    print(f"  [PASS] Exact P(B=True | J=T, M=T) = {post[True]:.5f}")

    # 5. Bayesian Network Likelihood Weighting --------------------------------
    print("Testing BayesianNetwork likelihood weighting...")
    approx = bn.likelihood_weighting("B", {"J": True, "M": True}, num_samples=15000, seed=42)
    assert abs(approx[True] - post[True]) < 0.03, (
        f"Approximate {approx[True]:.4f} too far from exact {post[True]:.4f}"
    )
    print(f"  [PASS] Likelihood Weighting P(B=True | J=T, M=T) = {approx[True]:.4f}")

    print("\nALL COURSE SCHEDULING & BAYESIAN NETWORK TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
