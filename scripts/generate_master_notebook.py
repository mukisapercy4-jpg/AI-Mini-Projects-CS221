"""Build the single, executable coursework notebook from assignment sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "ai_coursework_all_assignments.ipynb"

TEAM = """| # | Full name | Student / registration number | Email | Assignment / role |
|---|---|---|---|---|
| 1 | Ssenyondo Henry | 2500703882 | ssenyondohenry644@gmail.com | 1. Foundations |
| 2 | Ssewajje Jolly Hilton | 2500703588 / 25/U/03588/EVE | hiltonjolly@gmail.com | 2. Sentiment classification |
| 3 | Atwiine Precious Mwebaze | 2500703347 | preciousinshuti3@gmail.com | 3. Route planning |
| 4 | Opolot Emmanuel Marcus | 2500729501 | marcusemma480@gmail.com | 4. MountainCar |
| 5 | Kabanda Joel | 2500703381 | joelkabanda45@gmail.com | 4. MountainCar |
| 6 | Munanura Jill | 2500705824 | jillmunanura32@gmail.com | 5. Pac-Man |
| 7 | Mukisa Percy | 2500703468 / 25/U/03468/EVE | mukisapercy4@gmail.com | 5. Pac-Man |
| 8 | Joseph Mary Yiga | 2500703615 | josephmaryyiga@gmail.com | 6. Course scheduling and Bayesian networks |
| 9 | Muzira Joshua Kwizera | 2500703484 / 25/U/03484/EVE | joshngoboka1@gmail.com | 7. Car tracking and logic |
| 10 | Ariko Joseph Ojangole | 2500703330 / 25/U/03330/EVE | arikolojangole@gmail.com | Integration and submission lead |"""

ASSIGNMENTS = [
    ("01_foundations", "submission_foundations.py", "test_foundations.py", "run_foundations_experiments.py", "1. Foundations", "Sparse vectors, text helpers, regression gradients, and probability MLE."),
    ("02_sentiment", "submission_sentiment.py", "test_sentiment.py", "run_sentiment_experiments.py", "2. Sentiment Classification", "Word and character features, hinge-loss SGD, evaluation, and k-means."),
    ("03_route_planning", "submission_route.py", "test_route.py", "run_route_experiments.py", "3. Route Planning", "Uniform-cost search, A*, Euclidean distance, and landmark heuristics."),
    ("04_mountaincar", "submission_mountaincar.py", "test_mountaincar.py", "run_mountaincar_experiments.py", "4. Controlling MountainCar", "Discretized value iteration, tile coding, and Q-learning."),
    ("05_pacman", "submission_pacman.py", "test_pacman.py", "run_pacman_experiments.py", "5. Pac-Man", "Reflex, minimax, alpha-beta, expectimax, and state evaluation."),
    ("06_course_scheduling", "submission_scheduling.py", "test_scheduling.py", "run_scheduling_experiments.py", "6. Course Scheduling and Bayesian Networks", "CSP modeling, backtracking, MRV/LCV, AC-3, and probabilistic inference."),
    ("07_car_tracking", "submission_car_tracking.py", "test_car_tracking.py", "run_car_tracking_experiments.py", "7. Car Tracking and Logic", "Exact HMM inference, particle filtering, CNF conversion, and resolution."),
]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> None:
    cells = [
        markdown(
            "# Makerere University - Department of Computer Science\n"
            "## Artificial Intelligence Mini-Project Coursework\n\n"
            "**Course:** CS221-aligned AI coursework  \n"
            "**Submission deadline:** 5 December 2026  \n"
            "**Required programming deliverable:** this single executable notebook\n\n"
            "### Team and assignment roles\n\n" + TEAM +
            "\n\n### How to run\n\n"
            "Run this notebook from top to bottom. Each assignment implementation is included in its own code cell, followed by its local verification and experiment cells. The setup cell locates the project folder and makes the supplied assignment engines available."
        ),
        code(
            "from pathlib import Path\n"
            "import os, runpy, sys\n"
            "ROOT = next((p.resolve() for p in (Path.cwd(), *Path.cwd().parents) if (p / 'assignments').is_dir()), None)\n"
            "if ROOT is None:\n    raise FileNotFoundError('Open this notebook from inside the coursework project folder.')\n"
            "for folder in sorted((ROOT / 'assignments').iterdir()):\n"
            "    if folder.is_dir():\n        sys.path.insert(0, str(folder))\n"
            "print(f'Project folder: {ROOT}')\n"
            "print('Assignment support files are ready.')"
        ),
    ]
    for folder, filename, test_filename, experiment_filename, title, summary in ASSIGNMENTS:
        source_path = ROOT / "assignments" / folder / filename
        test_path = ROOT / "assignments" / folder / test_filename
        experiment_path = ROOT / "assignments" / folder / experiment_filename
        cells.append(markdown(f"---\n## {title}\n\n{summary}\n\n**Implementation source:** `{source_path.relative_to(ROOT).as_posix()}`"))
        cells.append(code(source_path.read_text(encoding="utf-8")))
        if test_path.is_file():
            cells.append(markdown("### Local verification\n\nRun the assignment's assertion-based checks."))
            cells.append(code(f"runpy.run_path(str(ROOT / {test_path.relative_to(ROOT).as_posix()!r}), run_name='__main__')"))
        if experiment_path.is_file():
            cells.append(markdown("### Experiment and results\n\nRun the supplied deterministic experiment or benchmark and display its measured results."))
            runner = "run_benchmarks" if folder in {"01_foundations", "05_pacman"} else "run_experiments"
            call = "experiment['run_benchmarks'](num_trials=3)" if folder == "05_pacman" else f"experiment[{runner!r}]()"
            cells.append(code(
                f"experiment = runpy.run_path(str(ROOT / {experiment_path.relative_to(ROOT).as_posix()!r}))\n"
                f"{call}"
            ))
    cells.append(markdown(
        "---\n## Integration summary\n\n"
        "The seven implementation modules and their local checks are included above. "
        "A successful full run should leave no code cell in an error state. Reported measurements are generated by the supplied experiment scripts and depend on the execution environment."
    ))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {NOTEBOOK_PATH.relative_to(ROOT)} with {len(cells)} cells.")


if __name__ == "__main__":
    build()
