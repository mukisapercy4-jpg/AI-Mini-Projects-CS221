import json

with open("notebooks/ai_coursework_all_assignments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Section 5 cells
cells = nb["cells"]
new_cells = []

for cell in cells:
    if cell.get("cell_type") == "markdown" and "5. Assignment 5: Pac-Man" in "".join(cell.get("source", [])):
        # Insert detailed Markdown header for Assignment 5
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "## 5. Assignment 5: Pac-Man Multi-Agent Adversarial Search (`pacman`)\n",
                "**Authors:** Mukisa Percy & Jill  \n",
                "**Topics Covered:** Reflex Agents, Adversarial Minimax, Multi-Ghost Alpha-Beta Pruning, Stochastic Expectimax, Linear Feature Evaluation Functions.\n",
                "\n",
                "### Overview & Problem Modeling\n",
                "Pac-Man operates in a discrete 2D grid world. The objective is to maximize score by consuming food pellets and capsules while avoiding lethal collisions with adversarial ghosts. We model the problem as a multi-agent extensive-form game:\n",
                "- **Pac-Man (Agent 0):** Maximizing player (MAX).\n",
                "- **Ghosts (Agents 1, ..., k):** Minimizing players (MIN) in adversarial search, or stochastic players in Expectimax.\n",
                "- **Ply:** Consists of 1 turn for Pac-Man followed by 1 turn for each ghost."
            ]
        })
    elif cell.get("cell_type") == "code" and "ASSIGNMENT 5: PAC-MAN MULTI-AGENT ADVERSARIAL SEARCH" in "".join(cell.get("source", [])):
        # Replace stub with complete implementation code cells
        with open("assignments/05_pacman/pacman_engine.py", "r", encoding="utf-8") as f_eng:
            engine_code = f_eng.read()
        with open("assignments/05_pacman/submission_pacman.py", "r", encoding="utf-8") as f_sub:
            sub_code = f_sub.read()

        # Cell 1: Engine and GameState
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# -----------------------------------------------------------------------------\n",
                "# 5.1 PAC-MAN GRID WORLD ENGINE & GAME STATE REPRESENTATION\n",
                "# -----------------------------------------------------------------------------\n",
                engine_code
            ]
        })

        # Cell 2: Search Agents (Reflex, Minimax, AlphaBeta, Expectimax, betterEval)
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# -----------------------------------------------------------------------------\n",
                "# 5.2 MULTI-AGENT SEARCH ALGORITHMS (MINIMAX, ALPHA-BETA, EXPECTIMAX)\n",
                "# -----------------------------------------------------------------------------\n",
                sub_code
            ]
        })

        # Cell 3: Empirical Benchmarks & Visualization
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.3 Experimental Evaluation & Benchmarking\n",
                "We benchmark `ReflexAgent`, `MinimaxAgent`, `AlphaBetaAgent`, and `ExpectimaxAgent` (using `betterEvaluationFunction`) over identical game seeds."
            ]
        })
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# -----------------------------------------------------------------------------\n",
                "# 5.4 BENCHMARK SIMULATION AND COMPARATIVE PLOTS\n",
                "# -----------------------------------------------------------------------------\n",
                "import time\n",
                "\n",
                "class BenchmarkGhost:\n",
                "    def __init__(self, index: int):\n",
                "        self.index = index\n",
                "    def getAction(self, state):\n",
                "        actions = state.getLegalActions(self.index)\n",
                "        return random.choice(actions) if actions else 'Stop'\n",
                "\n",
                "def evaluate_agent(agent, num_trials=3):\n",
                "    scores = []\n",
                "    wins = 0\n",
                "    times = []\n",
                "    for t in range(num_trials):\n",
                "        random.seed(42 + t)\n",
                "        s = PacmanGameState(width=8, height=6, pacman_pos=(1, 1), ghost_positions=[(6, 4)], capsules=[(6, 1)])\n",
                "        ghost = BenchmarkGhost(1)\n",
                "        t0 = time.time()\n",
                "        steps = 0\n",
                "        while not s.isWin() and not s.isLose() and steps < 100:\n",
                "            a = agent.getAction(s)\n",
                "            s = s.generateSuccessor(0, a)\n",
                "            steps += 1\n",
                "            if s.isWin() or s.isLose():\n",
                "                break\n",
                "            ga = ghost.getAction(s)\n",
                "            s = s.generateSuccessor(1, ga)\n",
                "        elapsed = (time.time() - t0) * 1000.0 / max(1, steps)\n",
                "        scores.append(s.getScore())\n",
                "        times.append(elapsed)\n",
                "        if s.isWin():\n",
                "            wins += 1\n",
                "    return (wins / num_trials) * 100.0, np.mean(scores), np.mean(times)\n",
                "\n",
                "agent_instances = {\n",
                "    'Reflex': ReflexAgent(),\n",
                "    'Minimax (d=2)': MinimaxAgent(depth=2),\n",
                "    'AlphaBeta (d=2)': AlphaBetaAgent(depth=2),\n",
                "    'Expectimax (d=2)': ExpectimaxAgent(evalFn=betterEvaluationFunction, depth=2)\n",
                "}\n",
                "\n",
                "names, win_rates, avg_scores, avg_latencies = [], [], [], []\n",
                "print(f\"{'Agent':<20} | {'Win Rate (%)':<15} | {'Avg Score':<12} | {'Move Latency (ms)':<15}\")\n",
                "print('-' * 70)\n",
                "for name, ag in agent_instances.items():\n",
                "    wr, scr, lat = evaluate_agent(ag, num_trials=3)\n",
                "    names.append(name)\n",
                "    win_rates.append(wr)\n",
                "    avg_scores.append(scr)\n",
                "    avg_latencies.append(lat)\n",
                "    print(f\"{name:<20} | {wr:<15.1f} | {scr:<12.1f} | {lat:<15.2f}\")\n",
                "\n",
                "# Plotting comparison\n",
                "fig, ax1 = plt.subplots(figsize=(9, 4.5))\n",
                "color = 'tab:blue'\n",
                "ax1.set_xlabel('Search Agent Architecture')\n",
                "ax1.set_ylabel('Average Game Score', color=color)\n",
                "bars = ax1.bar(names, avg_scores, color=color, alpha=0.6, width=0.4, label='Avg Score')\n",
                "ax1.tick_params(axis='y', labelcolor=color)\n",
                "ax1.grid(axis='y', linestyle='--', alpha=0.7)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "color = 'tab:red'\n",
                "ax2.set_ylabel('Win Rate (%)', color=color)\n",
                "ax2.plot(names, win_rates, color=color, marker='o', linewidth=2.5, label='Win Rate %')\n",
                "ax2.tick_params(axis='y', labelcolor=color)\n",
                "ax2.set_ylim(-10, 110)\n",
                "\n",
                "plt.title('Pac-Man Search Algorithms: Performance & Score Comparison (Percy & Jill)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        })
    else:
        new_cells.append(cell)

nb["cells"] = new_cells
with open("notebooks/ai_coursework_all_assignments.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Master notebook Section 5 successfully upgraded!")
