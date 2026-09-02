#!/usr/bin/env python3
"""Build an immutable 256K or 440K mixed 10-short + 5-long suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets/LongBench/data"
COHORTS = HERE / "datasets/LongBench/cohorts"
SUITES = HERE / "datasets/LongBench/suites"
CORE = HERE / "benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
DEFAULT_MODEL = Path(
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/"
    "DeepSeek-V4-Flash-0731"
)
GEOMETRIES = {
    "256K": {
        "target_tokens": 262144,
        "chunk_tokens": 32768,
        "short_profile": "256k_10q/hotpotqa/profile.json",
        "long_source": "256k_3q_long30",
        "long_rows": {
            "long_hotpotqa_2": ("hotpotqa", (6, 0)),
            "long_musique_2": ("musique", (0, 3)),
            "long_multifieldqa_en_1": ("multifieldqa_en", (0,)),
        },
        "selection_origin": "posthoc_long_output_showcase_20260830",
    },
    "440K": {
        "target_tokens": 450560,
        "chunk_tokens": 56320,
        "short_profile": "440k_10q/hotpotqa/profile.json",
        "long_source": "440k_3q_long30",
        "long_rows": {
            "long_hotpotqa_2": ("hotpotqa", (0, 4)),
            "long_musique_2": ("musique", (0, 4)),
            "long_multifieldqa_en_1": ("multifieldqa_en", (0,)),
        },
        "selection_origin": "output_blind_frozen_long30_440k",
    },
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different frozen file: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _load_rows(dataset: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{dataset}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _derive_profile(
    *,
    model: Path,
    length: str,
    dataset: str,
    row_ids: tuple[int, ...],
    group_id: str,
) -> Path:
    geometry = GEOMETRIES[length]
    chunk_tokens = int(geometry["chunk_tokens"])
    target_tokens = int(geometry["target_tokens"])
    source_dir = COHORTS / str(geometry["long_source"]) / dataset
    source_profile = json.loads(
        (source_dir / "profile.json").read_text(encoding="utf-8")
    )
    source_data = json.loads(
        (source_dir / "data_selection.json").read_text(encoding="utf-8")
    )
    query_by_row = {
        int(entry["row_id"]): entry
        for entry in source_data["selection"]["queries"]
    }
    if not set(row_ids).issubset(query_by_row):
        raise ValueError(f"{dataset} selected rows are absent from source profile")
    payload = dict(source_data)
    payload.pop("selection_sha256", None)
    payload["geometry"] = dict(payload["geometry"])
    payload["geometry"]["num_queries"] = len(row_ids)
    payload["selection"] = dict(payload["selection"])
    payload["selection"]["queries"] = [
        {**query_by_row[row_id], "query_index": query_index}
        for query_index, row_id in enumerate(row_ids)
    ]
    selection = dict(payload)
    selection["selection_sha256"] = _canonical_sha256(payload)

    destination = COHORTS / f"{length.lower()}_suite15" / group_id
    data_path = destination / "data_selection.json"
    prompt_path = destination / "prompt_manifest.json"
    profile_path = destination / "profile.json"
    with tempfile.TemporaryDirectory(prefix="redknot-suite15-") as tmp:
        temporary = Path(tmp)
        temporary_data = temporary / "data_selection.json"
        temporary_prompt = temporary / "prompt_manifest.json"
        temporary_data.write_bytes(_json_bytes(selection))
        os.environ.update(
            {
                "REDKNOT_MODEL_PATH": str(model.resolve()),
                "REDKNOT_LONGBENCH_DIR": str(DATA_DIR.resolve()),
                "REDKNOT_DATASETS": dataset,
                "REDKNOT_ENGINE_MODE": "indexer_hot",
                "REDKNOT_IH_NUM_CHUNKS": "8",
                "REDKNOT_IH_CHUNK_TOKENS": str(chunk_tokens),
                "REDKNOT_IH_NUM_QUERIES": str(len(row_ids)),
                "REDKNOT_IH_PURE_PROMPT_MODE": "official_rag_v1",
                "REDKNOT_IH_LONG_OUTPUT_TOKENS": "30",
                "REDKNOT_IH_EXPECTED_DATASET": dataset,
                "REDKNOT_THINKING_MODE": "chat",
                "REDKNOT_REASONING_EFFORT": "low",
            }
        )
        namespace = runpy.run_path(str(CORE), run_name=f"suite15_{group_id}")
        tokenizer = namespace["_ih_load_tokenizer"]()
        chunks, queries, replay = namespace["_ih_load"](
            tokenizer,
            chunk_tokens,
            8,
            len(row_ids),
            row_offset=0,
            manifest_path=str(temporary_data),
            exclude_manifest_paths=(),
            return_manifest=True,
            dataset_name=dataset,
            dataset_dir=str(DATA_DIR.resolve()),
        )
        builder = namespace["_ih_build_official_pure_prompt"]
        globals_ = builder.__globals__
        globals_["IH_EXPECTED_DATA_SELECTION_SHA256"] = selection[
            "selection_sha256"
        ]
        globals_["IH_EXPECTED_QUERY_ROW_IDS"] = row_ids
        globals_["IH_EXPECTED_DATASET"] = dataset
        globals_["IH_PROMPT_MANIFEST"] = ""
        globals_["IH_PROMPT_MANIFEST_OUT"] = str(temporary_prompt)
        prompt_chunks, rebuilt_queries, prompt = builder(
            tokenizer, chunks, queries, replay, chunk_tokens=chunk_tokens
        )
        if len(prompt_chunks) != 8 or any(
            len(chunk) != chunk_tokens for chunk in prompt_chunks
        ):
            raise AssertionError("suite prompt geometry drifted")
        if len(rebuilt_queries) != len(row_ids):
            raise AssertionError("suite prompt query count drifted")
        cases = prompt.get("cases")
        if cases is None:
            source = prompt["source"]
            geometry = prompt["geometry"]
            frozen = prompt["prompt"]
            cases = [
                {
                    "query_index": 0,
                    "query_row_id": int(source["query_row_id"]),
                    "expected_source_chunk": int(source["expected_source_chunk"]),
                    "online_suffix_tokens": int(geometry["online_suffix_tokens"]),
                    "total_tokens": int(geometry["total_tokens"]),
                    "text_sha256": str(frozen["text_sha256"]),
                    "full_input_ids_sha256": str(frozen["full_input_ids_sha256"]),
                    "online_suffix_hash": str(frozen["online_suffix_hash"]),
                    "question_sha256": str(frozen["question_sha256"]),
                    "answers_sha256": str(frozen["answers_sha256"]),
                }
            ]
        profile = {
            "format": "redknot_multidataset_profile_v2",
            "dataset": dataset,
            "target_tokens": target_tokens,
            "output_blind": True,
            "long_output_target_tokens": 30,
            "answer_style": "direct_answer_plus_document_evidence_v1",
            "cohort_index": int(source_profile["cohort_index"]),
            "row_offset": int(source_profile["row_offset"]),
            "exclude_manifests": list(source_profile["exclude_manifests"]),
            "num_queries": len(row_ids),
            "num_chunks": 8,
            "chunk_tokens": chunk_tokens,
            "query_start": target_tokens,
            "query_row_ids": list(row_ids),
            "data_manifest": str(data_path.resolve()),
            "data_selection_sha256": selection["selection_sha256"],
            "prompt_manifest": str(prompt_path.resolve()),
            "prompt_manifest_sha256": prompt["prompt_manifest_sha256"],
            "max_total_tokens": max(int(case["total_tokens"]) for case in cases),
            "offline_chunk_hashes": prompt["prompt"]["offline_chunk_hashes"],
            "cases": cases,
        }
        _publish_exact(data_path, temporary_data.read_bytes())
        _publish_exact(prompt_path, temporary_prompt.read_bytes())
        _publish_exact(profile_path, _json_bytes(profile))
    return profile_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--length", choices=tuple(GEOMETRIES), default="256K")
    args = parser.parse_args()
    length = str(args.length)
    geometry = GEOMETRIES[length]
    suite_format = f"redknot_deepseek_v4_flash_{length.lower()}_case_v1"
    suite_name = f"redknot_deepseek_v4_flash_{length.lower()}_15case_v1"
    model = args.model.expanduser().resolve()
    if not (model / "tokenizer.json").is_file():
        raise FileNotFoundError(f"checkpoint tokenizer is absent: {model}")

    standard_profile_path = COHORTS / str(geometry["short_profile"])
    long_specs = tuple(
        (group_id, dataset, row_ids)
        for group_id, (dataset, row_ids) in geometry["long_rows"].items()
    )
    long_profiles = {
        group_id: _derive_profile(
            model=model,
            length=length,
            dataset=dataset,
            row_ids=row_ids,
            group_id=group_id,
        )
        for group_id, dataset, row_ids in long_specs
    }
    records: list[dict[str, Any]] = []

    def append_cases(
        *,
        group_id: str,
        dataset: str,
        profile_path: Path,
        selection_origin: str,
        eligible: bool,
    ) -> None:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        rows = _load_rows(dataset)
        relative_profile = os.path.relpath(profile_path, SUITES)
        for case in profile["cases"]:
            row_id = int(case["query_row_id"])
            row = rows[row_id]
            target = int(profile.get("long_output_target_tokens", 0))
            records.append(
                {
                    "format": suite_format,
                    "suite": suite_name,
                    "suite_case_index": len(records),
                    "case_id": f"{group_id}-row{row_id}",
                    "group_id": group_id,
                    "length": length,
                    "dataset": dataset,
                    "query_index": int(case["query_index"]),
                    "query_row_id": row_id,
                    "question": str(row["input"]),
                    "answers": [str(value) for value in row["answers"]],
                    "answer_style": (
                        "direct_answer_plus_document_evidence_v1"
                        if target
                        else "shortest_exact_span_v1"
                    ),
                    "target_output_tokens": target,
                    "full_input_ids_sha256": str(case["full_input_ids_sha256"]),
                    "profile": relative_profile,
                    "selection_origin": selection_origin,
                    "eligible_for_accuracy_aggregate": eligible,
                }
            )

    append_cases(
        group_id="short_hotpotqa_10",
        dataset="hotpotqa",
        profile_path=standard_profile_path,
        selection_origin=f"output_blind_standard_{length.lower()}",
        eligible=True,
    )
    for group_id, dataset, _ in long_specs:
        append_cases(
            group_id=group_id,
            dataset=dataset,
            profile_path=long_profiles[group_id],
            selection_origin=str(geometry["selection_origin"]),
            eligible=False,
        )
    if len(records) != 15:
        raise AssertionError(f"suite builder produced {len(records)} cases")
    destination = SUITES / f"release_{length.lower()}_15case.jsonl"
    content = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in records
    )
    _publish_exact(destination, content)
    _publish_exact(
        destination.with_suffix(".jsonl.sha256"),
        (hashlib.sha256(content).hexdigest() + "  " + destination.name + "\n").encode(),
    )
    print(
        json.dumps(
            {
                "suite": str(destination),
                "sha256": hashlib.sha256(content).hexdigest(),
                "cases": len(records),
                "groups": 4,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
