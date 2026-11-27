"""
Unit Test Suite for Assignment 7: Car Tracking (HMM & Particle Filtering) & Logic
Author: Muzira Joshua Kwizera (Student No. 2500703484)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from car_tracking_engine import CarGridWorld
from submission_car_tracking import (
    ExactInference,
    ParticleFilter,
    Atom,
    Not,
    And,
    Or,
    Implies,
    Equiv,
    to_cnf,
    resolution_entails,
)


def run_tests():
    # 1. Exact HMM Inference ---------------------------------------------------
    print("Testing ExactInference (Forward HMM)...")
    world = CarGridWorld(rows=6, cols=6, sensor_std=0.8, seed=42)
    traj = world.simulate_trajectory(num_steps=10, start=(1, 1), observer=(0, 0))
    exact = ExactInference(6, 6, world.transition_probs, sensor_std=0.8)
    for step, (true_pos, dist) in enumerate(traj):
        exact.observe(0, 0, dist)
        if step < len(traj) - 1:
            exact.elapse_time()
    belief = exact.get_belief()
    assert abs(sum(belief.values()) - 1.0) < 1e-6, "Belief must sum to 1.0"
    est = exact.map_estimate()
    final_true = traj[-1][0]
    manhattan_err = abs(est[0] - final_true[0]) + abs(est[1] - final_true[1])
    assert manhattan_err <= 3, f"Exact HMM estimate {est} too far from {final_true}"
    print(f"  [PASS] ExactInference MAP={est}, True={final_true}, Err={manhattan_err}")

    # 2. Particle Filter -------------------------------------------------------
    print("Testing ParticleFilter...")
    pf = ParticleFilter(6, 6, world.transition_probs, sensor_std=0.8, num_particles=600, seed=42)
    for step, (true_pos, dist) in enumerate(traj):
        pf.observe(0, 0, dist)
        if step < len(traj) - 1:
            pf.elapse_time()
    pf_belief = pf.get_belief()
    assert abs(sum(pf_belief.values()) - 1.0) < 1e-6, "Particle belief must sum to 1.0"
    pf_est = pf.map_estimate()
    pf_err = abs(pf_est[0] - final_true[0]) + abs(pf_est[1] - final_true[1])
    assert pf_err <= 4, f"ParticleFilter estimate {pf_est} too far from {final_true}"
    print(f"  [PASS] ParticleFilter MAP={pf_est}, True={final_true}, Err={pf_err}")

    # 3. CNF Conversion --------------------------------------------------------
    print("Testing Propositional Logic to_cnf...")
    P, Q, R = Atom("P"), Atom("Q"), Atom("R")
    # (P -> Q) <-> (~P | Q)
    clauses = to_cnf(Implies(P, Q))
    assert clauses == {frozenset({"~P", "Q"})}, f"Unexpected CNF: {clauses}"
    # ~(P & Q) -> (~P | ~Q)
    clauses2 = to_cnf(Not(And(P, Q)))
    assert clauses2 == {frozenset({"~P", "~Q"})}, f"Unexpected CNF: {clauses2}"
    print("  [PASS] CNF transformation verified.")

    # 4. Resolution Entailment -------------------------------------------------
    print("Testing resolution_entails (Modus Ponens & Contrapositive)...")
    kb = [Implies(P, Q), Implies(Q, R), P]
    assert resolution_entails(kb, R) is True, "KB should entail R"
    assert resolution_entails(kb, Not(R)) is False, "KB must not entail ~R"
    # Unrelated atom S should not be entailed
    S = Atom("S")
    assert resolution_entails(kb, S) is False, "KB must not entail unrelated S"
    print("  [PASS] Resolution theorem prover verified.")

    print("\nALL CAR TRACKING & LOGIC TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
