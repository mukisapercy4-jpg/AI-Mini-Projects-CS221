"""
MountainCar Continuous Physical Environment Engine
Simulates the underpowered car climbing a sinusoidal valley for CS221 Assignment 4.
"""

import math
import random
from typing import Tuple, Dict, Any, List

class MountainCarEnv:
    """
    Standard MountainCar physical dynamics.
    State: (position x, velocity v)
    Actions: 0 = Push Left (-1), 1 = Coast (0), 2 = Push Right (+1)
    """
    MIN_POSITION = -1.2
    MAX_POSITION = 0.6
    MAX_SPEED = 0.07
    GOAL_POSITION = 0.5

    ACTIONS = [-1, 0, 1]  # Force multipliers

    def __init__(self, seed: int = 42):
        self.seed(seed)
        self.state = (-0.5, 0.0)

    def seed(self, s: int) -> None:
        random.seed(s)

    def reset(self) -> Tuple[float, float]:
        """Resets car randomly near valley bottom."""
        x = -0.6 + random.uniform(0.0, 0.2)
        v = 0.0
        self.state = (round(x, 4), round(v, 4))
        return self.state

    def step(self, action_idx: int) -> Tuple[Tuple[float, float], float, bool, Dict[str, Any]]:
        """
        Takes action and applies gravity and engine force.
        Returns: (next_state, reward, done, info)
        """
        x, v = self.state
        force = self.ACTIONS[action_idx]

        # Physics transition
        # v_{t+1} = v_t + 0.001 * a - 0.0025 * cos(3 * x_t)
        v_next = v + 0.001 * force - 0.0025 * math.cos(3.0 * x)
        v_next = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_next))

        x_next = x + v_next
        x_next = max(self.MIN_POSITION, min(self.MAX_POSITION, x_next))

        # Reset velocity to 0 if car hits left wall
        if x_next == self.MIN_POSITION and v_next < 0:
            v_next = 0.0

        done = bool(x_next >= self.GOAL_POSITION)
        reward = 0.0 if done else -1.0

        self.state = (round(x_next, 4), round(v_next, 4))
        return self.state, reward, done, {}
