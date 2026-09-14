"""Read the implementation index without importing either inference engine."""

import json
from importlib.resources import files
from typing import Any


def implementation_map() -> dict[str, Any]:
    """Return detached documentation metadata, not a runtime capability probe."""
    return json.loads(
        files("vllm_redknot")
        .joinpath("implementation_map.json")
        .read_text(encoding="utf-8")
    )
