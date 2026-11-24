"""
MountainCar Experiment & Benchmark Runner
Trains Q-learning agent across multiple episodes and evaluates policy success rate.
"""

import time
from mountaincar_engine import MountainCarEnv
from submission_mountaincar import (
    DiscretizedMountainCarMDP,
    value_iteration_mountaincar,
    TileFeatureExtractor,
    QLearningAgent
)

def run_experiments():
    print("=" * 65)
    print("CS221 ASSIGNMENT 4: MOUNTAINCAR BENCHMARKS & EXPERIMENTS")
    print("=" * 65)

    # 1. Value Iteration on Discretized MDP
    print("[EXPERIMENT 1] Discretized Value Iteration Benchmark:")
    mdp = DiscretizedMountainCarMDP(x_bins=12, v_bins=12)
    t0 = time.perf_counter()
    V = value_iteration_mountaincar(mdp, gamma=0.99, num_iters=25)
    dt_vi = (time.perf_counter() - t0) * 1000.0
    
    val_valley = V[mdp.discretize_state(-0.5, 0.0)]
    print(f"  Grid size: 144 states | 25 Iterations in {dt_vi:.2f} ms")
    print(f"  Value at valley bottom (-0.5, 0.0): {val_valley:.3f}")

    # 2. Q-Learning Training Loop
    print("\n[EXPERIMENT 2] Training Q-Learning Agent with Tile Coding:")
    env = MountainCarEnv(seed=42)
    extractor = TileFeatureExtractor(num_tilings=4, bins_per_tile=8)
    agent = QLearningAgent(extractor, alpha=0.08, gamma=0.99, epsilon=0.15)

    num_episodes = 60
    max_steps_per_episode = 200
    episode_rewards = []

    t_rl = time.perf_counter()
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        
        # Decay epsilon
        agent.epsilon = max(0.02, 0.20 * (0.96 ** ep))

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        if (ep + 1) % 15 == 0:
            avg_rew = sum(episode_rewards[-15:]) / 15.0
            print(f"  Episode: {ep+1:3d} | Avg Reward (last 15): {avg_rew:7.1f} | Epsilon: {agent.epsilon:.3f}")

    rl_time = (time.perf_counter() - t_rl) * 1000.0
    print(f"\n[SUMMARY] Trained {num_episodes} episodes in {rl_time:.2f} ms | Learned {len(agent.weights)} active tile weights")
    print("=" * 65)

if __name__ == "__main__":
    run_experiments()
