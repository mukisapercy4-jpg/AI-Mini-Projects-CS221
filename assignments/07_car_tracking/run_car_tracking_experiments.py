"""
Experiment & Benchmark Runner — Assignment 7: Car Tracking & Logic
Author: Muzira Joshua Kwizera (Student No. 2500703484)
"""

import time
from car_tracking_engine import CarGridWorld
from submission_car_tracking import (
    ExactInference,
    ParticleFilter,
    Atom,
    Not,
    And,
    Implies,
    resolution_entails,
)


def run_experiments():
    print("=" * 65)
    print("CS221 ASSIGNMENT 7: CAR TRACKING (HMM) & LOGIC EXPERIMENTS")
    print("=" * 65)

    world = CarGridWorld(rows=8, cols=8, sensor_std=1.0, seed=42)
    traj = world.simulate_trajectory(num_steps=12, start=(1, 1), observer=(0, 0))

    # 1. Exact Inference Benchmark
    exact = ExactInference(8, 8, world.transition_probs, sensor_std=1.0)
    t0 = time.perf_counter()
    exact_errs = []
    for step, (true_pos, dist) in enumerate(traj):
        exact.observe(0, 0, dist)
        est = exact.map_estimate()
        exact_errs.append(abs(est[0] - true_pos[0]) + abs(est[1] - true_pos[1]))
        if step < len(traj) - 1:
            exact.elapse_time()
    exact_ms = (time.perf_counter() - t0) * 1000.0

    print("\n[CAR TRACKING BENCHMARK] 8x8 Grid, 12 Steps:")
    print(f"{'Method':<26} | {'Avg L1 Err':>10} | {'Final L1':>8} | {'Time (ms)':>9}")
    print("-" * 62)
    print(f"{'ExactInference (HMM)':<26} | {sum(exact_errs)/len(exact_errs):10.2f} | {exact_errs[-1]:8d} | {exact_ms:9.2f}")

    # 2. Particle Filter Scaling
    for n_p in [100, 500, 2000]:
        pf = ParticleFilter(8, 8, world.transition_probs, sensor_std=1.0, num_particles=n_p, seed=42)
        t1 = time.perf_counter()
        pf_errs = []
        for step, (true_pos, dist) in enumerate(traj):
            pf.observe(0, 0, dist)
            est = pf.map_estimate()
            pf_errs.append(abs(est[0] - true_pos[0]) + abs(est[1] - true_pos[1]))
            if step < len(traj) - 1:
                pf.elapse_time()
        pf_ms = (time.perf_counter() - t1) * 1000.0
        print(f"{f'ParticleFilter (N={n_p})':<26} | {sum(pf_errs)/len(pf_errs):10.2f} | {pf_errs[-1]:8d} | {pf_ms:9.2f}")

    # 3. Logic Resolution Demo
    print("\n[LOGIC RESOLUTION] Automated Reasoning Demo:")
    Rain = Atom("Rain")
    Wet = Atom("Wet")
    Slippery = Atom("Slippery")
    kb = [Implies(Rain, Wet), Implies(Wet, Slippery), Rain]
    print(f"  KB: Rain -> Wet, Wet -> Slippery, Rain")
    print(f"  Entails Slippery?  {resolution_entails(kb, Slippery)}")
    print(f"  Entails ~Slippery? {resolution_entails(kb, Not(Slippery))}")

    print("=" * 65)
    print("Assignment 7 experiments completed successfully.")


if __name__ == "__main__":
    run_experiments()
