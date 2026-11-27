"""
CS221 Assignment 3: Route Planning Implementation
Author: Precious Atwiine (Group AI Mini-Projects)
"""

import math
import heapq
from typing import List, Tuple, Dict, Any, Optional, Callable
from route_engine import CityMapGraph

class SearchProblem:
    """Abstract interface for a general state-space search problem."""
    def start_state(self) -> Any:
        """Return the initial state of the search problem."""
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        """Return whether the supplied state satisfies the goal test."""
        raise NotImplementedError

    def successors_and_costs(self, state: Any) -> List[Tuple[str, Any, float]]:
        """Returns list of (action, next_state, edge_cost)."""
        raise NotImplementedError

class CityRouteProblem(SearchProblem):
    """Concrete Route Planning problem on a road graph network."""
    def __init__(self, graph: CityMapGraph, start: str, goal: str):
        """Initialize this object's state and configuration."""
        self.graph = graph
        self.start = start
        self.goal = goal

    def start_state(self) -> str:
        """Return the initial state of the search problem."""
        return self.start

    def is_end(self, state: str) -> bool:
        """Return whether the supplied state satisfies the goal test."""
        return state == self.goal

    def successors_and_costs(self, state: str) -> List[Tuple[str, str, float]]:
        """Return available actions, successor states, and step costs."""
        results = []
        for nbr, cost in self.graph.get_neighbors(state):
            action = f"Travel to {nbr}"
            results.append((action, nbr, cost))
        return results

# =============================================================================
# SEARCH ALGORITHMS
# =============================================================================

class SearchResult:
    """Store a search path, its actions, accumulated cost, and expansion count."""
    def __init__(self, path: List[str], actions: List[str], total_cost: float, num_expanded: int):
        """Initialize this object's state and configuration."""
        self.path = path
        self.actions = actions
        self.total_cost = total_cost
        self.num_expanded = num_expanded

def uniform_cost_search(problem: SearchProblem) -> Optional[SearchResult]:
    """
    Uniform Cost Search (Dijkstra's algorithm variant).
    Explores nodes in increasing order of accumulated path cost g(s).
    """
    # Priority Queue elements: (cost, counter, state, path, actions)
    counter = 0
    start = problem.start_state()
    frontier = [(0.0, counter, start, [start], [])]
    explored: Dict[Any, float] = {}
    num_expanded = 0

    while frontier:
        cost, _, state, path, actions = heapq.heappop(frontier)

        if state in explored and explored[state] <= cost:
            continue
        explored[state] = cost
        num_expanded += 1

        if problem.is_end(state):
            return SearchResult(path, actions, cost, num_expanded)

        for act, next_s, step_cost in problem.successors_and_costs(state):
            new_cost = cost + step_cost
            if next_s not in explored or new_cost < explored[next_s]:
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, next_s, path + [next_s], actions + [act]))

    return None

def a_star_search(
    problem: SearchProblem,
    heuristic: Callable[[Any], float]
) -> Optional[SearchResult]:
    """
    A* Search Algorithm with arbitrary heuristic function h(s).
    Explores nodes ordered by priority f(s) = g(s) + h(s).
    """
    counter = 0
    start = problem.start_state()
    h_start = heuristic(start)
    # Priority Queue elements: (f_score, g_score, counter, state, path, actions)
    frontier = [(h_start, 0.0, counter, start, [start], [])]
    g_scores: Dict[Any, float] = {start: 0.0}
    num_expanded = 0

    while frontier:
        f, g, _, state, path, actions = heapq.heappop(frontier)

        # Check if already reached via cheaper g-score
        if g > g_scores.get(state, float("inf")):
            continue
        num_expanded += 1

        if problem.is_end(state):
            return SearchResult(path, actions, g, num_expanded)

        for act, next_s, step_cost in problem.successors_and_costs(state):
            new_g = g + step_cost
            if new_g < g_scores.get(next_s, float("inf")):
                g_scores[next_s] = new_g
                h_val = heuristic(next_s)
                new_f = new_g + h_val
                counter += 1
                heapq.heappush(frontier, (new_f, new_g, counter, next_s, path + [next_s], actions + [act]))

    return None

# =============================================================================
# HEURISTICS
# =============================================================================

def euclidean_heuristic(graph: CityMapGraph, goal: str) -> Callable[[str], float]:
    """Euclidean straight-line distance heuristic to goal."""
    gx, gy = graph.get_coordinates(goal)
    def h(state: str) -> float:
        """Estimate straight-line distance from a state to the goal."""
        x, y = graph.get_coordinates(state)
        return math.hypot(x - gx, y - gy)
    return h

class LandmarkHeuristic:
    """
    Landmark (Differential) Heuristic:
    Precomputes exact shortest path distances from a selected landmark node L to all nodes.
    By the triangle inequality: h_L(s) = |d(s, L) - d(goal, L)|
    Guaranteed to be consistent and admissible.
    """
    def __init__(self, graph: CityMapGraph, landmark: str, goal: str):
        """Initialize this object's state and configuration."""
        self.graph = graph
        self.landmark = landmark
        self.goal = goal
        self.dist_to_landmark = self._compute_all_distances(landmark)

    def _compute_all_distances(self, source: str) -> Dict[str, float]:
        """Compute shortest-path distances from the landmark to reachable graph nodes."""
        dists: Dict[str, float] = {source: 0.0}
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dists[u]:
                continue
            for v, w in self.graph.get_neighbors(u):
                if d + w < dists.get(v, float("inf")):
                    dists[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        return dists

    def __call__(self, state: str) -> float:
        """Return the landmark lower-bound estimate for the supplied state."""
        d_state = self.dist_to_landmark.get(state, 0.0)
        d_goal = self.dist_to_landmark.get(self.goal, 0.0)
        return abs(d_state - d_goal)
