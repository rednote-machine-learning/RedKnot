#!/usr/bin/env python3
"""Build an explicitly posthoc, non-evaluation showcase from a completed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=5)
    args = parser.parse_args()
    if args.per_group <= 0:
        raise ValueError("--per-group must be positive")

    summary = args.summary.expanduser().resolve()
    cohort_root = args.cohort_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    run = json.loads(summary.read_text(encoding="utf-8"))
    groups = []
    for record in run.get("runs", []):
        pairs = record.get("output_comparisons", [])
        by_query: dict[int, list[dict[str, object]]] = {}
        for pair in pairs:
            by_query.setdefault(int(pair["query_index"]), []).append(pair)
        eligible = [
            query_index
            for query_index, repeats in sorted(by_query.items())
            if repeats
            and all(
                str(item.get("dense_output", ""))
                == str(item.get("redknot_output", ""))
                for item in repeats
            )
        ]
        length_key = str(record["length"]).lower() + "_10q"
        profile_path = cohort_root / length_key / str(record["dataset"]) / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        selected = []
        for query_index in eligible[: args.per_group]:
            repeats = sorted(by_query[query_index], key=lambda item: int(item["repeat"]))
            selected.append(
                {
                    "query_index": query_index,
                    "query_row_id": int(profile["query_row_ids"][query_index]),
                    "question": str(repeats[0]["question"]),
                    "repeats": [
                        {
                            "repeat": int(item["repeat"]),
                            "dense_output": str(item["dense_output"]),
                            "redknot_output": str(item["redknot_output"]),
                        }
                        for item in repeats
                    ],
                }
            )
        groups.append(
            {
                "length": record["length"],
                "dataset": record["dataset"],
                "source_result": record["result"],
                "source_result_sha256": record["result_sha256"],
                "source_profile": str(profile_path),
                "source_profile_sha256": _sha256(profile_path),
                "primary_pair_count": len(pairs),
                "primary_exact_pair_count": sum(
                    str(item.get("dense_output", ""))
                    == str(item.get("redknot_output", ""))
                    for item in pairs
                ),
                "eligible_query_count": len(eligible),
                "selected_cases": selected,
            }
        )

    showcase = {
        "format": "redknot_posthoc_showcase_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_bias": "posthoc",
        "not_for_aggregate_metrics": True,
        "primary_evaluation_is_authoritative": True,
        "selection_policy": (
            "After the full output-blind matrix completed, select at most "
            f"{args.per_group} lowest-query-index cases per length/dataset for "
            "which every recorded repeat has byte-identical dense and RedKnot "
            "output. No selected case is used in aggregate accuracy or speed metrics."
        ),
        "source_summary": str(summary),
        "source_summary_sha256": _sha256(summary),
        "groups": groups,
    }
    _atomic_json(output, showcase)
    output.with_suffix(output.suffix + ".sha256").write_text(
        _sha256(output) + "  " + output.name + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
