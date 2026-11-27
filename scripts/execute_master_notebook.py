"""Execute the coursework notebook with Python's standard library only.

This mirrors the sequential code-cell behavior of a Python Jupyter kernel and
saves stdout and execution counts back into the notebook. The cells in this
project use plain Python and do not require IPython display hooks.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "ai_coursework_all_assignments.ipynb"


def execute() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0
    os.chdir(ROOT)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        source = "".join(cell.get("source", []))
        output = io.StringIO()
        print(f"[{execution_count}] Running code cell {index + 1}/{len(notebook['cells'])}...", flush=True)
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(source, f"<notebook-cell-{index + 1}>", "exec"), namespace)
        except BaseException as exc:
            cell["execution_count"] = execution_count
            cell["outputs"] = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output.getvalue().splitlines(keepends=True),
                },
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                },
            ]
            NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        text = output.getvalue()
        cell["execution_count"] = execution_count
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}]
                           if text else [])
        if text:
            print(text, end="", flush=True)

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Executed {execution_count} code cells successfully and saved their outputs.")


if __name__ == "__main__":
    execute()
