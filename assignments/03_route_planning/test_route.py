"""
Unit Test Suite for Assignment 3: Route Planning
Verifies UCS and A* search correctness, optimal path cost, and heuristic consistency.
"""

from route_engine import create_sample_city_network
from submission_route import (
    CityRouteProblem,
    uniform_cost_search,
    a_star_search,
    euclidean_heuristic,
    LandmarkHeuristic
)

def run_tests():
    print("Testing City Map Network Graph...")
    graph = create_sample_city_network()
    assert "Downtown" in graph.nodes
    assert len(graph.get_neighbors("Downtown")) >= 3
    print("  [PASS] City map graph initialized.")

    problem = CityRouteProblem(graph, start="Downtown", goal="Airport")

    print("Testing Uniform Cost Search (UCS)...")
    res_ucs = uniform_cost_search(problem)
    assert res_ucs is not None
    assert res_ucs.path[0] == "Downtown"
    assert res_ucs.path[-1] == "Airport"
    print(f"  [PASS] UCS found path: {' -> '.join(res_ucs.path)} with cost: {res_ucs.total_cost:.2f} (Expanded: {res_ucs.num_expanded})")

    print("Testing A* Search with Euclidean Heuristic...")
    h_euc = euclidean_heuristic(graph, goal="Airport")
    res_astar = a_star_search(problem, h_euc)
    assert res_astar is not None
    assert abs(res_astar.total_cost - res_ucs.total_cost) < 1e-6, "A* must find optimal path cost"
    print(f"  [PASS] A* (Euclidean) found optimal path with cost: {res_astar.total_cost:.2f} (Expanded: {res_astar.num_expanded})")
    assert res_astar.num_expanded <= res_ucs.num_expanded, "A* must expand fewer or equal nodes than UCS"

    print("Testing Landmark Heuristic Admissibility & Consistency...")
    h_landmark = LandmarkHeuristic(graph, landmark="Harbor", goal="Airport")
    res_lm = a_star_search(problem, h_landmark)
    assert res_lm is not None
    assert abs(res_lm.total_cost - res_ucs.total_cost) < 1e-6
    print(f"  [PASS] A* (Landmark) found optimal path with cost: {res_lm.total_cost:.2f} (Expanded: {res_lm.num_expanded})")

    # Verify consistency along graph edges: h(u) <= cost(u, v) + h(v)
    for u in graph.nodes:
        for v, cost in graph.get_neighbors(u):
            hu = h_landmark(u)
            hv = h_landmark(v)
            assert hu <= cost + hv + 1e-6, f"Landmark heuristic inconsistency: {hu} > {cost} + {hv}"
    print("  [PASS] Landmark heuristic proven consistent on all edges.")

    print("\nALL BASIC ROUTE PLANNING TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
