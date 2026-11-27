"""
Car Tracking 2D Grid Simulator — Assignment 7
Simulates a moving car on a discrete (rows x cols) grid and generates noisy
Euclidean distance observations from an observer's position.
"""

import math
import random
from typing import Dict, List, Tuple


class CarGridWorld:
    def __init__(self, rows: int = 8, cols: int = 8, sensor_std: float = 1.0, seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.sensor_std = sensor_std
        self.rng = random.Random(seed)
        self.transition_probs = self._build_transitions()

    def _build_transitions(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
        """Each cell transitions mostly East/South/adjacent to simulate structured road motion."""
        trans: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors: List[Tuple[int, int]] = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neighbors.append((nr, nc))
                # Favor moving right/down if available, otherwise uniform
                weights = {}
                for nr, nc in neighbors:
                    if (nr - r, nc - c) in [(0, 1), (1, 0)]:
                        weights[(nr, nc)] = 3.0
                    else:
                        weights[(nr, nc)] = 1.0
                total = sum(weights.values())
                trans[(r, c)] = {k: v / total for k, v in weights.items()}
        return trans

    def simulate_trajectory(
        self, num_steps: int = 15, start: Tuple[int, int] = (1, 1), observer: Tuple[int, int] = (0, 0)
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Returns list of ((true_row, true_col), noisy_distance_from_observer)."""
        pos = start
        trajectory = []
        for _ in range(num_steps):
            true_dist = math.hypot(pos[0] - observer[0], pos[1] - observer[1])
            noisy_dist = max(0.0, true_dist + self.rng.gauss(0.0, self.sensor_std))
            trajectory.append((pos, noisy_dist))
            # Sample next position
            dist_map = self.transition_probs[pos]
            r_val = self.rng.random()
            cum = 0.0
            for nxt, p in dist_map.items():
                cum += p
                if r_val <= cum:
                    pos = nxt
                    break
        return trajectory
