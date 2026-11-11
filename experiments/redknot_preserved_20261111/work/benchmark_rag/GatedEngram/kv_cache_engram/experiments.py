"""Deterministic CPU stress experiments for the admission logic, NOT model eval.

Run with ``python -B -m kv_cache_engram.experiments --output NEW_DIRECTORY``.
No model/checkpoint, network, subprocess, Torch or accelerator is used. Artificial
error patterns test detection/fallback and expose missed unprobed changes. Their
thresholds and reuse fractions must not be deployed as model-calibrated policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time

from .cache import CacheArtifact
from .core import CacheIdentity, digest, integer, token_digest
from .gate import GatePolicy, make_plan
from .reuse_audit import PairedKV, audit_pair

SCENARIOS = ("identical", "head127_only", "propagated_boundary1024",
             "interior_span2048", "fixed_sparse_spikes", "adversarial_unprobed_spike")


def synthetic_pair(name, *, token_count=32768):
    """Regenerable token geometry; values are artificial, not text embeddings."""
    integer(token_count, "token_count", 2048)
    if name not in SCENARIOS:
        raise ValueError("unknown synthetic scenario")
    ordinal = SCENARIOS.index(name)
    document = tuple(1000 + ordinal * token_count + i for i in range(token_count))
    layer = (3, 3, 18, 9, 18, 18)[ordinal]  # Labels only; no actual layers executed.
    identity = CacheIdentity(digest("synthetic-model-not-DeepSeek"),
        digest("synthetic-tokenizer"), digest("synthetic-cpu-v1"),
        token_digest(document), layer, token_count, 4)
    cached = tuple((1.0 + (i % 17) / 20.0, 0.75, -0.5, 1.25) for i in range(token_count))
    # Exactly eight logical chunks. For nonfirst contexts this document is last.
    offset = 0 if name == "identical" else 7 * token_count
    prefix = (42,) * offset
    suffix = (43,) * (8 * token_count - offset - token_count)
    request = prefix + document + suffix
    cache = CacheArtifact.build(identity, cached)
    policy = GatePolicy(enabled=True, block_size=512, probe_count=8,
        relative_tolerance=0.05, absolute_tolerance=0.05,
        calibration_id="synthetic-diagnostic-only-not-runtime-calibration")
    clean = PairedKV(cache, identity, document, request, offset, cached,
                    "synthetic-" + name, "synthetic")
    if name == "identical":
        changed = ()
    elif name == "head127_only":
        changed = tuple(range(127))
    elif name == "propagated_boundary1024":
        changed = tuple(range(1024))
    elif name == "interior_span2048":
        changed = tuple(range(token_count // 2, min(token_count, token_count // 2 + 2048)))
    elif name == "fixed_sparse_spikes":
        # Predetermined positions, independent of the gate's sampled positions.
        changed = tuple(range(257, token_count-128, 997))
    else:
        # Explicit adversarial counterexample, not a representative-data estimate.
        plan = make_plan(clean.binding(512, 1024), policy)
        covered = set(plan.probe_positions) | set(plan.mandatory_positions)
        changed = (next(p for p in plan.positions if p not in covered),)
    altered = set(changed)
    online = tuple((row[0] + 1.0, *row[1:]) if i in altered else row
                   for i, row in enumerate(cached))
    return PairedKV(cache, identity, document, request, offset, online,
                   clean.request_id, "synthetic"), policy, changed


def run_synthetic_suite(output, *, token_count=32768):
    integer(token_count, "token_count", 2048)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summaries = []
    for scenario in SCENARIOS:
        pair, policy, changed = synthetic_pair(scenario, token_count=token_count)
        report = audit_pair(pair, policy, oracle_relative_tolerance=0.05,
                            oracle_absolute_tolerance=0.05)
        report["synthetic_scenario"] = dict(name=scenario,
            changed_positions=list(changed), artificial_component_delta=1.0,
            adversarial=scenario == "adversarial_unprobed_spike",
            input_tokens=len(pair.request_token_ids), document_tokens=token_count,
            model_layers_executed=False, full_online_reference_is_synthetic=True)
        path = destination / (scenario + ".json.gz")
        with path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
                encoded = json.dumps(report, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()
                zipped.write(encoded)
        summaries.append(dict(scenario=scenario, layer_label=pair.online_identity.layer,
            source="synthetic", document_start=pair.document_start,
            input_sha256=pair.request_sha256, changed_rows=len(changed),
            summary=report["summary"], by_boundary=report["by_boundary"],
            policy=asdict(policy), full_report=path.name,
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(schema="kv-cache-engram-cpu-stress-experiment-v1",
        token_count_per_document=token_count, input_tokens=8*token_count,
        experiment_kind="synthetic-counterexamples-not-model-quality-evaluation",
        thresholds_fitted=False, empirical_model_calibration=False,
        gpu_experiments_started=0, actual_model_em_f1=None, actual_model_ttft=None,
        elapsed_cpu_workflow_s=time.perf_counter()-started, scenarios=summaries,
        interpretation=[
            "Boundary-row repair alone cannot establish deeper-context equivalence.",
            "A passing finite probe set cannot certify all unprobed rows; adversarial case demonstrates this.",
            "Fallback repeats probe work; report that overhead even when other blocks save rows.",
            "Do not choose model thresholds or acceptable EM/F1 loss from these synthetic numbers.",
            "Labels 3/9/18 are metadata examples, not measured differences between model layers.",
            "All scenarios retained. The full per-token/per-block reports are compressed alongside this summary."])
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory; never overwrites")
    parser.add_argument("--tokens", type=int, default=32768)
    args = parser.parse_args(argv)
    result = run_synthetic_suite(args.output, token_count=args.tokens)
    print(json.dumps({"kind": result["experiment_kind"],
                      "scenarios": [{"name": s["scenario"], **s["summary"]}
                                    for s in result["scenarios"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
