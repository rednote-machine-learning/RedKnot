#!/usr/bin/env python3
"""RedKnot qwen3 RAG entrypoint; default CPU migration plan, never implicit GPU."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.multimodel_rag import main

if __name__ == "__main__":
    raise SystemExit(main("qwen3"))
