#!/usr/bin/env python3
"""Freeze one output-blind 256K LongBench corpus and its prompt cases."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help=(
            "Previously frozen data-selection manifest to exclude. Repeat the "
            "flag to guarantee disjoint corpus/query rows across cohorts."
        ),
    )
    parser.add_argument("--data-out", required=True)
    parser.add_argument("--prompt-out", required=True)
    parser.add_argument("--profile-out", required=True)
    args = parser.parse_args()

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.row_offset < 0:
        raise ValueError("--row-offset must be non-negative")
    outputs = tuple(
        Path(value).expanduser().resolve()
        for value in (args.data_out, args.prompt_out, args.profile_out)
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite outputs: {outputs}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_manifests = tuple(
        str(Path(value).expanduser().resolve())
        for value in args.exclude_manifest
    )
    for path in exclusion_manifests:
        if not Path(path).is_file():
            raise FileNotFoundError(f"exclusion manifest is absent: {path}")

    os.environ.update(
        {
            "REDKNOT_MODEL_PATH": str(Path(args.model).expanduser().resolve()),
            "REDKNOT_LONGBENCH_DIR": str(
                Path(args.data_dir).expanduser().resolve()
            ),
            "REDKNOT_DATASETS": args.dataset,
            "REDKNOT_ENGINE_MODE": "indexer_hot",
            "REDKNOT_IH_NUM_CHUNKS": "8",
            "REDKNOT_IH_CHUNK_TOKENS": "32768",
            "REDKNOT_IH_NUM_QUERIES": str(args.num_queries),
            "REDKNOT_IH_PURE_PROMPT_MODE": "official_rag_v1",
            "REDKNOT_IH_EXPECTED_DATASET": args.dataset,
            "REDKNOT_THINKING_MODE": "chat",
            "REDKNOT_REASONING_EFFORT": "low",
        }
    )
    namespace = runpy.run_path(
        str(Path(args.core).expanduser().resolve()),
        run_name="redknot_multidataset_manifest_builder",
    )
    tokenizer = namespace["_ih_load_tokenizer"]()
    chunks, queries, selection = namespace["_ih_load"](
        tokenizer,
        32768,
        8,
        args.num_queries,
        row_offset=args.row_offset,
        manifest_path="",
        exclude_manifest_paths=exclusion_manifests,
        return_manifest=True,
        dataset_name=args.dataset,
        dataset_dir=str(Path(args.data_dir).expanduser().resolve()),
    )
    namespace["_ih_write_data_manifest"](str(outputs[0]), selection)
    query_rows = tuple(
        int(entry["row_id"])
        for entry in selection["selection"]["queries"]
    )
    builder = namespace["_ih_build_official_pure_prompt"]
    builder_globals = builder.__globals__
    builder_globals["IH_EXPECTED_DATA_SELECTION_SHA256"] = selection[
        "selection_sha256"
    ]
    builder_globals["IH_EXPECTED_QUERY_ROW_IDS"] = query_rows
    builder_globals["IH_EXPECTED_DATASET"] = args.dataset
    builder_globals["IH_PROMPT_MANIFEST"] = ""
    builder_globals["IH_PROMPT_MANIFEST_OUT"] = str(outputs[1])
    prompt_chunks, rebuilt_queries, prompt_manifest = builder(
        tokenizer,
        chunks,
        queries,
        selection,
        chunk_tokens=32768,
    )
    max_total_tokens = max(
        8 * 32768 + len(query[1]) for query in rebuilt_queries
    )
    if max_total_tokens != prompt_manifest["geometry"]["max_total_tokens"]:
        raise AssertionError("prompt profile maximum length drifted")
    profile = {
        "format": "redknot_multidataset_256k_profile_v1",
        "dataset": args.dataset,
        "output_blind": True,
        "row_offset": args.row_offset,
        "exclude_manifests": list(exclusion_manifests),
        "num_queries": args.num_queries,
        "num_chunks": 8,
        "chunk_tokens": 32768,
        "query_start": 8 * 32768,
        "query_row_ids": list(query_rows),
        "data_manifest": str(outputs[0]),
        "data_selection_sha256": selection["selection_sha256"],
        "prompt_manifest": str(outputs[1]),
        "prompt_manifest_sha256": prompt_manifest[
            "prompt_manifest_sha256"
        ],
        "max_total_tokens": max_total_tokens,
        "offline_chunk_hashes": prompt_manifest["prompt"][
            "offline_chunk_hashes"
        ],
        "cases": prompt_manifest["cases"],
    }
    outputs[2].write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
