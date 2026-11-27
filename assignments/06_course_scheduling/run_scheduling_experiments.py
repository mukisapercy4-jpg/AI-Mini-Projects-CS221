"""
Experiment & Benchmark Runner — Assignment 6: Course Scheduling & Bayesian Networks
Author: Joseph Mary Yiga (Student No. 2500703615)
"""

import time
from scheduling_engine import get_sample_course_catalog, get_alarm_bayesian_network
from submission_scheduling import (
    CSP,
    BacktrackingSearch,
    create_course_scheduling_csp,
    BayesianNetwork,
)


def run_experiments():
    print("=" * 65)
    print("CS221 ASSIGNMENT 6: COURSE SCHEDULING & BAYESIAN EXPERIMENTS")
    print("=" * 65)

    # 1. CSP Solver Ablation (4-Queens / Course Scheduling)
    n = 6
    queens = CSP()
    for col in range(n):
        queens.add_variable(col, list(range(n)))
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            queens.add_binary_factor(
                c1, c2,
                lambda r1, r2, d=(c2 - c1): 1.0 if (r1 != r2 and abs(r1 - r2) != d) else 0.0
            )

    configs = [
        ("Plain Backtracking", False, False, False),
        ("MRV + LCV", True, True, False),
        ("MRV + LCV + AC-3", True, True, True),
    ]
    print("\n[CSP BENCHMARK] 6-Queens Solver Ablation:")
    print(f"{'Configuration':<22} | {'Assignments':>11} | {'Backtracks':>10} | {'Time (ms)':>9}")
    print("-" * 62)
    for name, mrv, lcv, ac3 in configs:
        solver = BacktrackingSearch(queens, mrv=mrv, lcv=lcv, ac3=ac3)
        t0 = time.perf_counter()
        sol = solver.solve()
        dt = (time.perf_counter() - t0) * 1000.0
        assert sol is not None
        print(f"{name:<22} | {solver.num_assignments:11d} | {solver.num_backtracks:10d} | {dt:9.2f}")

    # 2. Course Schedule
    catalog = get_sample_course_catalog()
    sched_csp = create_course_scheduling_csp(catalog)
    sched_solver = BacktrackingSearch(sched_csp, mrv=True, lcv=True, ac3=True)
    sched = sched_solver.solve()
    print("\n[COURSE SCHEDULE] Valid 4-Quarter Plan:")
    for q in catalog["quarters"]:
        q_courses = [c for c, qr in sched.items() if qr == q]
        print(f"  Quarter {q}: {', '.join(q_courses) if q_courses else '(none)'}")

    # 3. Bayesian Network Inference Comparison
    net_data = get_alarm_bayesian_network()
    bn = BayesianNetwork(net_data["nodes"], net_data["parents"], net_data["cpts"])
    exact = bn.exact_inference("B", {"J": True, "M": True})
    print("\n[BAYESIAN INFERENCE] Query P(Burglary=True | JohnCalls=T, MaryCalls=T):")
    print(f"  Exact (Enumeration):       {exact[True]:.5f}")
    for n_samples in [500, 2000, 10000]:
        approx = bn.likelihood_weighting("B", {"J": True, "M": True}, num_samples=n_samples, seed=42)
        err = abs(approx[True] - exact[True])
        print(f"  Likelihood Weighting ({n_samples:5d}): {approx[True]:.5f} (abs err: {err:.5f})")

    print("=" * 65)
    print("Assignment 6 experiments completed successfully.")


if __name__ == "__main__":
    run_experiments()
