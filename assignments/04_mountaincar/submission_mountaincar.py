"""
CS221 Assignment 4: Controlling MountainCar Implementation
Authors: Marcus & Co-Lead (Group AI Mini-Projects)
"""

import math
import random
import collections
from typing import List, Tuple, Dict, Any, Optional
from mountaincar_engine import MountainCarEnv

# =============================================================================
# DISCRETIZATION & VALUE ITERATION (MDP)
# =============================================================================

class DiscretizedMountainCarMDP:
    """
    Discretizes continuous (position, velocity) state space into discrete grid bins.
    """
    def __init__(self, x_bins: int = 15, v_bins: int = 15):
        self.x_bins = x_bins
        self.v_bins = v_bins
        self.env = MountainCarEnv()

    def discretize_state(self, x: float, v: float) -> Tuple[int, int]:
        """Maps continuous state (x, v) to discrete indices (ix, iv)."""
        x_norm = (x - self.env.MIN_POSITION) / (self.env.MAX_POSITION - self.env.MIN_POSITION)
        v_norm = (v + self.env.MAX_SPEED) / (2.0 * self.env.MAX_SPEED)

        ix = min(self.x_bins - 1, max(0, int(x_norm * self.x_bins)))
        iv = min(self.v_bins - 1, max(0, int(v_norm * self.v_bins)))
        return (ix, iv)

    def continuous_center(self, ix: int, iv: int) -> Tuple[float, float]:
        """Returns the continuous midpoint of a bin."""
        x_span = (self.env.MAX_POSITION - self.env.MIN_POSITION) / self.x_bins
        v_span = (2.0 * self.env.MAX_SPEED) / self.v_bins
        x = self.env.MIN_POSITION + (ix + 0.5) * x_span
        v = -self.env.MAX_SPEED + (iv + 0.5) * v_span
        return (x, v)

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        ix, _ = state
        x_thresh = (self.env.GOAL_POSITION - self.env.MIN_POSITION) / (self.env.MAX_POSITION - self.env.MIN_POSITION)
        return ix >= int(x_thresh * self.x_bins)

def value_iteration_mountaincar(
    mdp: DiscretizedMountainCarMDP,
    gamma: float = 0.99,
    num_iters: int = 30
) -> Dict[Tuple[int, int], float]:
    """
    Computes optimal value function V*(s) over discretized state grid using Bellman optimality updates.
    """
    V: Dict[Tuple[int, int], float] = {
        (ix, iv): 0.0 for ix in range(mdp.x_bins) for iv in range(mdp.v_bins)
    }

    env = MountainCarEnv()
    for _ in range(num_iters):
        new_V: Dict[Tuple[int, int], float] = {}
        for ix in range(mdp.x_bins):
            for iv in range(mdp.v_bins):
                state = (ix, iv)
                if mdp.is_terminal(state):
                    new_V[state] = 0.0
                    continue

                best_val = -float("inf")
                cx, cv = mdp.continuous_center(ix, iv)
                
                for act_idx in range(3):
                    env.state = (cx, cv)
                    next_s, reward, done, _ = env.step(act_idx)
                    next_disc = mdp.discretize_state(next_s[0], next_s[1])
                    val = reward + (0.0 if done else gamma * V[next_disc])
                    if val > best_val:
                        best_val = val
                new_V[state] = best_val
        V = new_V
    return V

# =============================================================================
# Q-LEARNING WITH LINEAR FUNCTION APPROXIMATION
# =============================================================================

class TileFeatureExtractor:
    """
    Multi-resolution tiling feature extractor for continuous MountainCar state.
    """
    def __init__(self, num_tilings: int = 4, bins_per_tile: int = 8):
        self.num_tilings = num_tilings
        self.bins = bins_per_tile

    def extract_features(self, state: Tuple[float, float], action_idx: int) -> Dict[str, float]:
        x, v = state
        x_norm = (x - (-1.2)) / (0.6 - (-1.2))
        v_norm = (v + 0.07) / 0.14

        features: Dict[str, float] = collections.defaultdict(float)
        # Bias feature
        features[f"bias_a{action_idx}"] = 1.0

        for t in range(self.num_tilings):
            offset = t / (self.num_tilings * self.bins)
            bx = min(self.bins - 1, max(0, int((x_norm + offset) * self.bins)))
            bv = min(self.bins - 1, max(0, int((v_norm + offset) * self.bins)))
            features[f"tile_{t}_{bx}_{bv}_a{action_idx}"] = 1.0

        return dict(features)

class QLearningAgent:
    """
    Semi-gradient Q-learning agent with linear function approximation: Q_hat(s, a; w) = w . phi(s, a).
    """
    def __init__(
        self,
        feature_extractor: TileFeatureExtractor,
        alpha: float = 0.05,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.feature_extractor = feature_extractor
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights: Dict[str, float] = collections.defaultdict(float)

    def get_q_value(self, state: Tuple[float, float], action_idx: int) -> float:
        phi = self.feature_extractor.extract_features(state, action_idx)
        return sum(self.weights[k] * v for k, v in phi.items())

    def choose_action(self, state: Tuple[float, float], explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        q_vals = [self.get_q_value(state, a) for a in range(3)]
        max_q = max(q_vals)
        best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(
        self,
        state: Tuple[float, float],
        action: int,
        reward: float,
        next_state: Tuple[float, float],
        done: bool
    ) -> float:
        """
        Semi-gradient TD update:
        TD target = r + gamma * max_a' Q(s', a') (if not done)
        TD error = target - Q(s, a)
        w <- w + alpha * TD_error * phi(s, a)
        """
        q_current = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            q_next_max = max(self.get_q_value(next_state, a) for a in range(3))
            target = reward + self.gamma * q_next_max

        td_error = target - q_current
        phi = self.feature_extractor.extract_features(state, action)
        
        for k, v in phi.items():
            self.weights[k] += self.alpha * td_error * v

        return td_error
