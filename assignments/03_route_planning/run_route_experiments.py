"""
Route Planning Experiment & Benchmark Runner
Compares UCS vs A* across multiple search queries, evaluating node expansion reduction.
"""

import time
from route_engine import create_sample_city_network
from submission_route import (
    CityRouteProblem,
    uniform_cost_search,
    a_star_search,
    euclidean_heuristic,
    LandmarkHeuristic
)

def run_experiments():
    print("=" * 65)
    print("CS221 ASSIGNMENT 3: ROUTE PLANNING EXPERIMENTS")
    print("=" * 65)

    graph = create_sample_city_network()
    test_queries = [
        ("Downtown", "Airport"),
        ("Harbor", "Suburbs"),
        ("WestEnd", "EastGate"),
        ("Downtown", "Uptown"),
    ]

    print(f"{'Query':<22} | {'Algorithm':<14} | {'Cost':<6} | {'Expanded':<8} | {'Latency':<8}")
    print("-" * 65)

    for start, goal in test_queries:
        problem = CityRouteProblem(graph, start, goal)
        query_str = f"{start} -> {goal}"

        # 1. UCS
        t0 = time.perf_counter()
        res_ucs = uniform_cost_search(problem)
        dt_ucs = (time.perf_counter() - t0) * 1000.0
        print(f"{query_str:<22} | {'UCS (Dijkstra)':<14} | {res_ucs.total_cost:6.2f} | {res_ucs.num_expanded:8d} | {dt_ucs:6.2f} ms")

        # 2. A* (Euclidean)
        h_euc = euclidean_heuristic(graph, goal)
        t0 = time.perf_counter()
        res_a_euc = a_star_search(problem, h_euc)
        dt_euc = (time.perf_counter() - t0) * 1000.0
        print(f"{'':<22} | {'A* (Euclidean)':<14} | {res_a_euc.total_cost:6.2f} | {res_a_euc.num_expanded:8d} | {dt_euc:6.2f} ms")

        # 3. A* (Landmark)
        h_lm = LandmarkHeuristic(graph, landmark="Harbor", goal=goal)
        t0 = time.perf_counter()
        res_a_lm = a_star_search(problem, h_lm)
        dt_lm = (time.perf_counter() - t0) * 1000.0
        print(f"{'':<22} | {'A* (Landmark)':<14} | {res_a_lm.total_cost:6.2f} | {res_a_lm.num_expanded:8d} | {dt_lm:6.2f} ms")
        print("-" * 65)

    print("Route Planning experiments completed successfully.")

if __name__ == "__main__":
    run_experiments()
