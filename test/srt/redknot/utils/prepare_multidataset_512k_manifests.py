#!/usr/bin/env python3
"""Freeze one output-blind long-context LongBench corpus and prompt cohort."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path


DEFAULT_CHUNK_TOKENS = 65536
DEFAULT_NUM_CHUNKS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--cohort-index", type=int, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument(
        "--long-output-tokens",
        type=int,
        choices=(0, 30, 50),
        default=0,
        help=(
            "Keep 0 for the unchanged shortest-span benchmark, or freeze an "
            "isolated 30/50-token direct-answer-plus-evidence prompt cohort."
        ),
    )
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--data-out", required=True)
    parser.add_argument("--prompt-out", required=True)
    parser.add_argument("--profile-out", required=True)
    args = parser.parse_args()

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.row_offset < 0 or args.cohort_index < 0:
        raise ValueError("row offset and cohort index must be non-negative")
    if args.chunk_tokens <= 0 or args.chunk_tokens % 128:
        raise ValueError("--chunk-tokens must be a positive multiple of 128")
    if args.num_chunks <= 0:
        raise ValueError("--num-chunks must be positive")
    chunk_tokens = int(args.chunk_tokens)
    num_chunks = int(args.num_chunks)
    target_tokens = chunk_tokens * num_chunks
    outputs = tuple(
        Path(value).expanduser().resolve()
        for value in (args.data_out, args.prompt_out, args.profile_out)
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite outputs: {outputs}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    exclusions = tuple(
        str(Path(value).expanduser().resolve())
        for value in args.exclude_manifest
    )
    for path in exclusions:
        if not Path(path).is_file():
            raise FileNotFoundError(f"exclusion manifest is absent: {path}")

    model = str(Path(args.model).expanduser().resolve())
    data_dir = str(Path(args.data_dir).expanduser().resolve())
    os.environ.update(
        {
            "REDKNOT_MODEL_PATH": model,
            "REDKNOT_LONGBENCH_DIR": data_dir,
            "REDKNOT_DATASETS": args.dataset,
            "REDKNOT_ENGINE_MODE": "indexer_hot",
            "REDKNOT_IH_NUM_CHUNKS": str(num_chunks),
            "REDKNOT_IH_CHUNK_TOKENS": str(chunk_tokens),
            "REDKNOT_IH_NUM_QUERIES": str(args.num_queries),
            "REDKNOT_IH_PURE_PROMPT_MODE": "official_rag_v1",
            "REDKNOT_IH_LONG_OUTPUT_TOKENS": str(args.long_output_tokens),
            "REDKNOT_IH_EXPECTED_DATASET": args.dataset,
            "REDKNOT_THINKING_MODE": "chat",
            "REDKNOT_REASONING_EFFORT": "low",
        }
    )
    namespace = runpy.run_path(
        str(Path(args.core).expanduser().resolve()),
        run_name="redknot_multidataset_512k_manifest_builder",
    )
    tokenizer = namespace["_ih_load_tokenizer"]()
    chunks, queries, selection = namespace["_ih_load"](
        tokenizer,
        chunk_tokens,
        num_chunks,
        args.num_queries,
        row_offset=args.row_offset,
        manifest_path="",
        exclude_manifest_paths=exclusions,
        return_manifest=True,
        dataset_name=args.dataset,
        dataset_dir=data_dir,
    )
    namespace["_ih_write_data_manifest"](str(outputs[0]), selection)
    query_rows = tuple(
        int(entry["row_id"])
        for entry in selection["selection"]["queries"]
    )
    builder = namespace["_ih_build_official_pure_prompt"]
    globals_ = builder.__globals__
    globals_["IH_EXPECTED_DATA_SELECTION_SHA256"] = selection[
        "selection_sha256"
    ]
    globals_["IH_EXPECTED_QUERY_ROW_IDS"] = query_rows
    globals_["IH_EXPECTED_DATASET"] = args.dataset
    globals_["IH_PROMPT_MANIFEST"] = ""
    globals_["IH_PROMPT_MANIFEST_OUT"] = str(outputs[1])
    prompt_chunks, rebuilt_queries, prompt_manifest = builder(
        tokenizer,
        chunks,
        queries,
        selection,
        chunk_tokens=chunk_tokens,
    )
    if len(prompt_chunks) != num_chunks or any(
        len(chunk) != chunk_tokens for chunk in prompt_chunks
    ):
        raise AssertionError("long-context prompt geometry drifted")
    max_total_tokens = max(
        target_tokens + len(query[1]) for query in rebuilt_queries
    )
    manifest_max_total_tokens = prompt_manifest["geometry"].get(
        "max_total_tokens", prompt_manifest["geometry"].get("total_tokens")
    )
    if max_total_tokens != manifest_max_total_tokens:
        raise AssertionError("prompt profile maximum length drifted")
    cases = prompt_manifest.get("cases")
    if cases is None:
        source = prompt_manifest["source"]
        prompt = prompt_manifest["prompt"]
        geometry = prompt_manifest["geometry"]
        cases = [
            {
                "query_index": 0,
                "query_row_id": int(source["query_row_id"]),
                "expected_source_chunk": int(source["expected_source_chunk"]),
                "online_suffix_tokens": int(geometry["online_suffix_tokens"]),
                "total_tokens": int(geometry["total_tokens"]),
                "text_sha256": str(prompt["text_sha256"]),
                "full_input_ids_sha256": str(
                    prompt["full_input_ids_sha256"]
                ),
                "online_suffix_hash": str(prompt["online_suffix_hash"]),
                "question_sha256": str(prompt["question_sha256"]),
                "answers_sha256": str(prompt["answers_sha256"]),
            }
        ]
    if len(cases) != args.num_queries:
        raise AssertionError("prompt case count differs from the frozen cohort")
    profile = {
        "format": "redknot_multidataset_profile_v2",
        "dataset": args.dataset,
        "target_tokens": target_tokens,
        "output_blind": True,
        "long_output_target_tokens": int(args.long_output_tokens),
        "answer_style": (
            "direct_answer_plus_document_evidence_v1"
            if args.long_output_tokens
            else "shortest_exact_span_v1"
        ),
        "cohort_index": args.cohort_index,
        "row_offset": args.row_offset,
        "exclude_manifests": list(exclusions),
        "num_queries": args.num_queries,
        "num_chunks": num_chunks,
        "chunk_tokens": chunk_tokens,
        "query_start": target_tokens,
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
        "cases": cases,
    }
    outputs[2].write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
