"""CPU-only official-tokenizer preparation; never substitutes shorter inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--previous-manifest", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--official-encoding", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    sys.path.insert(0, str(args.experiment_root/"benchmark"))
    from prepare_rag256 import OfficialTokenizer, file_digest, load_source, seal, INSTRUCTION
    from prepare_attention_cohort import make_composite
    from pr148_native_runner import validate_manifest
    from gpu_pilot.pilot_protocol import build_source_plan, pilot_protocol, DATASETS
    from gpu_pilot.run_pilot import validate_inputs
    import pr148_native_runner as base
    if "torch" in sys.modules:
        raise RuntimeError("CPU preparation cannot import Torch")
    old = json.loads(args.previous_manifest.read_text())
    validate_manifest(old)
    sources, metadata = {}, []
    for dataset in DATASETS:
        info = next(item for item in old["sources"] if item["dataset"] == dataset)
        path = Path(info["path"])
        if file_digest(path) != info["sha256"]:
            raise ValueError("source identity changed: " + dataset)
        sources[dataset] = load_source(path)
        metadata.append(copy.deepcopy(info))
    t = old["tokenizer"]
    tokenizer = OfficialTokenizer(args.tokenizer, args.official_encoding,
                                  t["tokenizer_json_sha256"], t["encoding_sha256"])
    if tokenizer.identity != t:
        raise ValueError("official formatter/tokenizer mismatch")
    plan, protocol = build_source_plan(sources, [old]), pilot_protocol()
    rows = []
    for split, ordinal in (("calibration", 0), ("evaluation", 0), ("evaluation", 1)):
        for dataset in DATASETS:
            spec = plan["datasets"][dataset]["splits"][split]
            primary = spec["selected"][ordinal]
            row = make_composite(dataset, sources[dataset], primary["source_index"],
                "calibration" if split == "calibration" else "heldout", tokenizer,
                donors=spec["donor_source_indices"], anchor_chunk=primary["anchor_chunk"],
                partition_sha256=plan["sha256"], seed=protocol["seed"], chunk_tokens=32768, num_chunks=8)
            row.update(index=len(rows), sample_id=f"kce-pilot-v1:{split}:{dataset}:{row['source_id']}")
            rows.append(row)
    manifest = seal(dict(schema="redknot-deepinfer-real-rag-manifest-v1",
        artifact_kind="kv_cache_engram_native_pilot", tokenizer=tokenizer.identity,
        sources=metadata, warmup=rows[:2], cohort=rows[2:], selection_plan=plan,
        pilot_protocol_sha256=protocol["sha256"], num_chunks=8, chunk_tokens=32768,
        context_tokens=262144, instruction=INSTRUCTION,
        cache_protocol=copy.deepcopy(old.get("cache_protocol", {})),
        context_claim="true LongBench materials expanded to eight32K blocks, not native256K questions",
        source_exclusion_claim="declared exact-source exposure only, not semantic or training-set independence"))
    validate_inputs(base, manifest, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, obj in (("manifest.json", manifest), ("protocol.json", protocol), ("source-plan.json", plan)):
        with (args.output_dir/filename).open("x", encoding="utf-8") as out:
            json.dump(obj, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
    print(json.dumps(dict(status="prepared", gpu_initialized=False,
        manifest_sha256=manifest["sha256"], protocol_sha256=protocol["sha256"],
        rows=[dict(index=r["index"], dataset=r["dataset"], source_index=r["source_index"],
                   split=r["split"], input_tokens=r["input_tokens"], context_tokens=r["context_tokens"])
              for r in rows])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
