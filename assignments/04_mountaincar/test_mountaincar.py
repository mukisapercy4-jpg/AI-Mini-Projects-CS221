"""
Unit Test Suite for Assignment 4: MountainCar
Verifies physics engine dynamics, Value Iteration, and Q-Learning agent updates.
"""

from mountaincar_engine import MountainCarEnv
from submission_mountaincar import (
    DiscretizedMountainCarMDP,
    value_iteration_mountaincar,
    TileFeatureExtractor,
    QLearningAgent
)

def run_tests():
    print("Testing MountainCar Physical Simulator...")
    env = MountainCarEnv(seed=42)
    s = env.reset()
    assert -1.2 <= s[0] <= 0.6
    assert abs(s[1]) <= 0.07
    
    # Step right
    next_s, r, done, _ = env.step(action_idx=2)
    assert r == -1.0 or done
    print("  [PASS] Physics simulator step verified.")

    print("Testing Discretized MountainCar MDP & Value Iteration...")
    mdp = DiscretizedMountainCarMDP(x_bins=8, v_bins=8)
    disc = mdp.discretize_state(0.5, 0.0)
    assert mdp.is_terminal(disc)
    
    V_opt = value_iteration_mountaincar(mdp, gamma=0.95, num_iters=10)
    assert len(V_opt) == 64
    assert V_opt[disc] == 0.0
    print("  [PASS] Value iteration converged on discretized grid.")

    print("Testing Feature Extractor & Q-Learning Agent...")
    extractor = TileFeatureExtractor(num_tilings=3, bins_per_tile=6)
    phi = extractor.extract_features((-0.5, 0.0), action_idx=1)
    assert len(phi) >= 4
    assert phi["bias_a1"] == 1.0

    agent = QLearningAgent(extractor, alpha=0.1, gamma=0.99, epsilon=0.1)
    act = agent.choose_action((-0.5, 0.0), explore=False)
    assert act in [0, 1, 2]

    td_err = agent.update((-0.5, 0.0), 2, -1.0, (-0.49, 0.01), False)
    assert isinstance(td_err, float)
    print("  [PASS] Q-learning agent update verified.")

    print("\nALL BASIC MOUNTAINCAR TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
