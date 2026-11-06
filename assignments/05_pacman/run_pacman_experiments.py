"""
Pac-Man Multi-Agent Simulation and Experimentation Runner
Executes benchmark games comparing Reflex, Minimax, AlphaBeta, and Expectimax agents.
Assigned to: Mukisa Percy & Jill
"""

import time
import random
from typing import Dict, Any, List

from pacman_engine import PacmanGameState, Directions
from submission_pacman import (
    ReflexAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    ExpectimaxAgent,
    betterEvaluationFunction
)

class RandomGhost:
    def __init__(self, index: int):
        self.index = index

    def getAction(self, state: PacmanGameState) -> str:
        actions = state.getLegalActions(self.index)
        return random.choice(actions) if actions else Directions.STOP

def simulate_game(agent, ghost_type=RandomGhost, max_steps: int = 150, verbose: bool = False) -> Dict[str, Any]:
    state = PacmanGameState(
        width=8,
        height=6,
        pacman_pos=(1, 1),
        ghost_positions=[(6, 4)],
        capsules=[(6, 1)]
    )
    ghosts = [ghost_type(i) for i in range(1, state.getNumAgents())]

    start_time = time.time()
    step_count = 0
    move_times = []

    while not state.isWin() and not state.isLose() and step_count < max_steps:
        # Pacman move
        t0 = time.time()
        pacman_action = agent.getAction(state)
        move_times.append(time.time() - t0)

        state = state.generateSuccessor(0, pacman_action)
        step_count += 1
        if state.isWin() or state.isLose():
            break

        # Ghost moves
        for ghost in ghosts:
            g_action = ghost.getAction(state)
            state = state.generateSuccessor(ghost.index, g_action)
            if state.isWin() or state.isLose():
                break

    duration = time.time() - start_time
    return {
        "win": state.isWin(),
        "lose": state.isLose(),
        "score": state.getScore(),
        "steps": step_count,
        "duration": duration,
        "avg_move_time_ms": (sum(move_times) / len(move_times) * 1000.0) if move_times else 0.0
    }

def run_benchmarks(num_trials: int = 5):
    print("=" * 70)
    print("PAC-MAN MULTI-AGENT EXPERIMENTAL BENCHMARK")
    print("Makerere University AI Coursework — Mukisa Percy & Jill")
    print(f"Running {num_trials} trials per agent on standardized 8x6 layout...")
    print("=" * 70)

    agents = {
        "ReflexAgent": ReflexAgent(),
        "MinimaxAgent (d=2)": MinimaxAgent(depth=2),
        "AlphaBetaAgent (d=2)": AlphaBetaAgent(depth=2),
        "Expectimax (betterEval, d=2)": ExpectimaxAgent(evalFn=betterEvaluationFunction, depth=2)
    }

    results_table = []

    for name, agent in agents.items():
        print(f"\nEvaluating: {name}...")
        wins = 0
        total_score = 0.0
        total_steps = 0
        total_move_time = 0.0

        for trial in range(num_trials):
            random.seed(42 + trial)
            res = simulate_game(agent, max_steps=120)
            if res["win"]:
                wins += 1
            total_score += res["score"]
            total_steps += res["steps"]
            total_move_time += res["avg_move_time_ms"]
            outcome = "WIN " if res["win"] else ("LOSE" if res["lose"] else "TIMEOUT")
            print(f"  Trial {trial+1}: {outcome} | Score: {res['score']:.1f} | Steps: {res['steps']} | Move Time: {res['avg_move_time_ms']:.2f} ms")

        win_rate = (wins / num_trials) * 100.0
        avg_score = total_score / num_trials
        avg_steps = total_steps / num_trials
        avg_latency = total_move_time / num_trials

        results_table.append({
            "Agent": name,
            "Win Rate (%)": f"{win_rate:.1f}%",
            "Avg Score": f"{avg_score:.1f}",
            "Avg Steps": f"{avg_steps:.1f}",
            "Avg Latency (ms)": f"{avg_latency:.2f}"
        })

    print("\n" + "=" * 70)
    print(f"{'Agent':<30} | {'Win Rate':<10} | {'Avg Score':<10} | {'Avg Steps':<10} | {'Latency (ms)':<10}")
    print("-" * 70)
    for r in results_table:
        print(f"{r['Agent']:<30} | {r['Win Rate (%)']:<10} | {r['Avg Score']:<10} | {r['Avg Steps']:<10} | {r['Avg Latency (ms)']:<10}")
    print("=" * 70)
    print("\n[SUCCESS] Experimental evaluation complete!")

if __name__ == "__main__":
    run_benchmarks(num_trials=3)
