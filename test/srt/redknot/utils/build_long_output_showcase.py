#!/usr/bin/env python3
"""Build a transparent post-hoc long-output showcase from completed runs.

The output is intentionally ineligible for aggregate accuracy claims.  It is a
reproducible presentation artifact: every selected pair remains bound to the
source result, frozen prompt/input hash, dataset row and deterministic ranking
rule.  Standard output-blind benchmark profiles are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional


FORMAT = "redknot_long_output_posthoc_showcase_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direct_answer(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[0] if lines else ""


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.casefold()).strip()


def _readable(text: str) -> bool:
    value = str(text)
    return bool(value.strip()) and "\ufffd" not in value and not any(
        ord(char) < 32 and char not in "\n\r\t" for char in value
    )


def _candidates(
    result_path: Path,
    target: int,
    offline_context_tokens: Optional[int] = None,
) -> list[dict[str, Any]]:
    document = json.loads(result_path.read_text(encoding="utf-8"))
    config = document.get("config", {})
    configured_target = int(config.get("long_output_target_tokens", 0))
    if configured_target != target:
        return []
    configured_offline_tokens = int(config.get("chunk_tokens") or 0) * int(
        config.get("num_chunks") or 0
    )
    if (
        offline_context_tokens is not None
        and configured_offline_tokens != offline_context_tokens
    ):
        return []
    query_rows = list(config.get("query_row_ids") or [])
    rows = []
    result_sha = _sha256(result_path)
    for fallback_index, query in enumerate(document.get("queries") or []):
        query_index = int(query.get("query_index", fallback_index))
        dense = str(query.get("dense_text", ""))
        redknot = str(query.get("reuse_text", ""))
        dense_tokens = int(query.get("dense_output_tokens") or 0)
        redknot_tokens = int(query.get("reuse_output_tokens") or 0)
        dense_answer = _direct_answer(dense)
        redknot_answer = _direct_answer(redknot)
        first_line_match = bool(
            _normalize(dense_answer)
            and _normalize(dense_answer) == _normalize(redknot_answer)
        )
        readable = _readable(dense) and _readable(redknot)
        length_qualified = min(dense_tokens, redknot_tokens) >= target - 10
        cosine = float(query.get("min_top10_probability_cosine") or 0.0)
        agreement = float(query.get("min_generation_token_agreement") or 0.0)
        rows.append(
            {
                "dataset": str(config.get("dataset", "")),
                "context_tokens": int(config.get("full_input_tokens") or 0),
                "offline_context_tokens": configured_offline_tokens,
                "long_output_target_tokens": target,
                "query_index": query_index,
                "query_row_id": (
                    int(query_rows[query_index])
                    if 0 <= query_index < len(query_rows)
                    else None
                ),
                "question": str(query.get("question", "")),
                "dense_output": dense,
                "redknot_output": redknot,
                "dense_output_tokens": dense_tokens,
                "redknot_output_tokens": redknot_tokens,
                "dense_direct_answer": dense_answer,
                "redknot_direct_answer": redknot_answer,
                "direct_answer_match": first_line_match,
                "readable": readable,
                "length_qualified": length_qualified,
                "min_top10_probability_cosine": cosine,
                "min_generation_token_agreement": agreement,
                "full_input_ids_sha256": str(
                    query.get("full_input_ids_sha256", "")
                ),
                "prompt_manifest_sha256": str(
                    config.get("prompt_manifest_sha256", "")
                ),
                "source_result": str(result_path.resolve()),
                "source_result_sha256": result_sha,
                "_rank": (
                    int(first_line_match),
                    int(readable),
                    int(length_qualified),
                    cosine,
                    agreement,
                    -abs(dense_tokens - target) - abs(redknot_tokens - target),
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--target-output-tokens", type=int, choices=(30, 50), required=True
    )
    parser.add_argument(
        "--offline-context-tokens",
        type=int,
        choices=(262144, 450560),
        required=True,
        help="Freeze either the 256K or 440K eight-document cohort.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite showcase: {output}")
    result_paths: list[Path] = []
    for value in args.run_root:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"run root is absent: {root}")
        result_paths.extend(sorted(root.rglob("result.json")))
    candidates = [
        row
        for path in sorted(set(result_paths))
        for row in _candidates(
            path,
            args.target_output_tokens,
            args.offline_context_tokens,
        )
    ]
    qualified = [
        row
        for row in candidates
        if row["direct_answer_match"]
        and row["readable"]
        and row["length_qualified"]
    ]
    qualified.sort(
        key=lambda row: (
            tuple(-value for value in row["_rank"]),
            row["dataset"],
            row["context_tokens"],
            row["query_index"],
        )
    )
    selected = qualified[: args.limit]
    if len(selected) < args.limit:
        raise RuntimeError(
            f"only {len(selected)} qualifying cases exist; requested {args.limit}"
        )
    for row in candidates:
        row.pop("_rank", None)
    for row in selected:
        row.pop("_rank", None)
    payload = {
        "format": FORMAT,
        "selection_is_posthoc": True,
        "eligible_for_accuracy_aggregate": False,
        "purpose": "reproducible_long_output_presentation_only",
        "target_output_tokens": args.target_output_tokens,
        "offline_context_tokens": args.offline_context_tokens,
        "selection_rule": {
            "required": [
                "normalized_first_nonempty_line_matches",
                "both_outputs_are_readable",
                "both_output_lengths_are_at_least_target_minus_10",
            ],
            "ranking": [
                "min_top10_probability_cosine_desc",
                "min_generation_token_agreement_desc",
                "combined_target_length_error_asc",
                "dataset_context_query_stable_tiebreak",
            ],
        },
        "source_result_count": len(result_paths),
        "candidate_count": len(candidates),
        "qualifying_candidate_count": len(qualified),
        "selected": selected,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["showcase_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
