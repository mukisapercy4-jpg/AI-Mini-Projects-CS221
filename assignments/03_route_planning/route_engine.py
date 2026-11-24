"""
Route Planning Simulation Engine
Provides city map road graphs, waypoint coordinates, and search problem abstractions for CS221 Assignment 3.
"""

import math
from typing import List, Tuple, Dict, Any, Optional

class CityMapGraph:
    """
    Weighted directed graph representing a road transportation network.
    """
    def __init__(self):
        self.nodes: Dict[str, Tuple[float, float]] = {}  # node -> (x, y) coordinates
        self.edges: Dict[str, List[Tuple[str, float]]] = {}  # u -> [(v, cost)]

    def add_location(self, name: str, x: float, y: float) -> None:
        self.nodes[name] = (x, y)
        if name not in self.edges:
            self.edges[name] = []

    def add_road(self, u: str, v: str, cost: Optional[float] = None, bidirectional: bool = True) -> None:
        if cost is None:
            # Default to Euclidean distance
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            cost = math.hypot(x1 - x2, y1 - y2)
        
        self.edges[u].append((v, cost))
        if bidirectional:
            self.edges[v].append((u, cost))

    def get_coordinates(self, node: str) -> Tuple[float, float]:
        return self.nodes[node]

    def get_neighbors(self, node: str) -> List[Tuple[str, float]]:
        return self.edges.get(node, [])

def create_sample_city_network() -> CityMapGraph:
    """
    Creates a benchmark city map with multiple alternative routes, intersections, and landmarks.
    """
    g = CityMapGraph()
    # City locations
    g.add_location("Downtown", 0.0, 0.0)
    g.add_location("Uptown", 2.0, 5.0)
    g.add_location("WestEnd", -4.0, 2.0)
    g.add_location("EastGate", 5.0, 1.0)
    g.add_location("Harbor", -3.0, -4.0)
    g.add_location("Airport", 6.0, 6.0)
    g.add_location("Suburbs", 1.0, 8.0)
    g.add_location("TechPark", 4.0, 3.0)

    # Road network connections with distance costs
    g.add_road("Downtown", "WestEnd", 5.0)
    g.add_road("Downtown", "EastGate", 5.5)
    g.add_road("Downtown", "Harbor", 5.0)
    g.add_road("Downtown", "TechPark", 4.5)
    g.add_road("WestEnd", "Uptown", 7.0)
    g.add_road("EastGate", "TechPark", 2.5)
    g.add_road("TechPark", "Airport", 4.0)
    g.add_road("Uptown", "Suburbs", 3.2)
    g.add_road("Uptown", "Airport", 4.5)
    g.add_road("Suburbs", "Airport", 5.2)

    return g
