"""CPU counterexamples exercising calibration rejection, not model evaluation.

All policies and KV tolerances below are synthetic fixtures. They must not be
copied into deployment. Explicit adversarial spikes demonstrate blind spots;
their frequency is not an estimate of real-model anomaly prevalence.
"""
from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .cache import CacheArtifact
from .calibration_workflow import CalibrationExample, evaluate_frozen, fit_candidates
from .core import CacheIdentity, digest, token_digest
from .coverage import UNIFORM, miss_probability, plan_coverage, required_probes
from .gate import GatePolicy, make_plan
from .reuse_audit import PairedKV


def _example(index, split, changed=()):
    tokens = tuple(1000 + 256 * index + i for i in range(256))
    identity = CacheIdentity(digest("synthetic-v03-not-a-real-model"),
        digest("synthetic-tokenizer"), digest("synthetic-cpu-v03"),
        token_digest(tokens), 3, 256, 4)
    cached = ((1.0, 0.75, -0.5, 1.25),) * 256
    altered = set(changed)
    online = tuple((2.0, *row[1:]) if i in altered else row
                   for i, row in enumerate(cached))
    pair = PairedKV(CacheArtifact.build(identity, cached), identity, tokens,
                    (42, index) + tokens + (43,), 2, online,
                    "synthetic-request-" + str(index), "synthetic")
    return CalibrationExample(pair, "synthetic-source-" + str(index), split)


def _unprobed(example, policy):
    plan = make_plan(example.pair.binding(0, 256), policy)
    covered = set(plan.probe_positions) | set(plan.mandatory_positions)
    return next(p for p in plan.positions if p not in covered)


def run_calibration_experiment(output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    # Frozen BEFORE constructing counterexamples. This is adversarial testing,
    # not a representative holdout distribution or threshold optimization.
    candidates = tuple(GatePolicy(enabled=True, block_size=256, probe_count=count,
        head_guard=0, tail_guard=0, relative_tolerance=0.05, absolute_tolerance=0.05,
        calibration_id="synthetic-v03-diagnostic-not-runtime") for count in (8, 32, 128, 256))
    limits = dict(oracle_relative_tolerance=0.05, oracle_absolute_tolerance=0.05,
                  max_false_accept_blocks=0, max_bad_reused_rows=0)
    clean_fit = fit_candidates([_example(0, "calibration"), _example(1, "calibration")],
                                candidates, **limits)
    clean_validation = evaluate_frozen(clean_fit, [_example(2, "evaluation")])
    target = _example(3, "evaluation")
    spike = _unprobed(target, clean_fit.selected_policy)
    shifted_validation = evaluate_frozen(clean_fit, [_example(3, "evaluation", (spike,))])
    # Every finite-sample candidate receives a dedicated counterexample in the
    # SAME calibration cohort; no troublesome case is discarded per candidate.
    hard_examples = [_example(10, "calibration")]
    spikes = []
    for index, policy in enumerate(candidates[:-1], 11):
        position = _unprobed(_example(index, "calibration"), policy)
        hard_examples.append(_example(index, "calibration", (position,)))
        spikes.append(dict(source_index=index, position=position,
                           targeted_probe_count=policy.probe_count))
    hard_fit = fit_candidates(hard_examples, candidates, **limits)
    hard_validation = evaluate_frozen(hard_fit, [_example(20, "evaluation")])
    artifacts = {
        "clean_calibration": clean_fit.to_dict(),
        "clean_heldout": clean_validation,
        "adversarial_heldout": shifted_validation,
        "sparse_counterexample_calibration": hard_fit.to_dict(),
        "no_eligible_candidate_heldout": hard_validation,
    }
    files = {}
    for name, data in artifacts.items():
        path = destination / (name + ".json.gz")
        with path.open("xb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
                compressed.write(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode())
        files[name] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary = dict(schema="kv-cache-engram-cpu-calibration-experiment-v1",
        data_origin="synthetic", thresholds_for_fixture_only=True,
        actual_model_captures=0, gpu_experiments_started=0,
        actual_model_em_f1=None, actual_model_ttft=None, runtime_admission_created=False,
        candidates=[asdict(p) for p in candidates], limits=limits,
        clean_heldout=dict(selected_probe_count=clean_fit.selected_policy.probe_count,
                          kv_validation_pass=clean_validation["kv_validation_pass"]),
        adversarial_heldout=dict(position=spike,
            deliberately_unprobed=True, kv_validation_pass=shifted_validation["kv_validation_pass"],
            summary=shifted_validation["report"]["summary"],
            candidate_reselection_performed=shifted_validation["candidate_reselection_performed"]),
        sparse_calibration=dict(counterexamples=spikes, mode=hard_fit.mode,
            heldout_pass=hard_validation["kv_validation_pass"],
            candidates=[dict(probe_count=r["policy"]["probe_count"],
                             eligible=r["eligible"], reasons=r["rejection_reasons"],
                             summary=r["summary"])
                        for r in hard_fit.to_dict()["payload"]["candidate_reports"]]),
        uniform_single_bad_row_budget=dict(candidate_rows=512,
            eight_probe_miss_fraction=str(miss_probability(512, 1, 8)),
            eight_probe_miss_probability=float(miss_probability(512, 1, 8)),
            probes_for_one_percent_conditional_miss=required_probes(512, max_miss_probability=0.01),
            plan_with_budget_8=plan_coverage(512, available_probes=8,
                max_miss_probability=0.01, sampling_scheme=UNIFORM).to_dict()),
        artifacts=files,
        interpretation=[
            "Clean calibration is not evidence that isolated anomalous rows cannot occur.",
            "An adversarial heldout miss rejects the frozen candidate; it never retunes on heldout.",
            "All finite-probe candidates failing the declared budget means full recompute, not a best-effort pass.",
            "Full-row probing has no actual reuse and cannot qualify as a useful reuse candidate.",
            "Uniform coverage math does not apply to the deterministic gate used by these fixtures.",
            "No EM/F1, native capture, live gate admission, or inference speed was measured."])
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory, never overwrite")
    result = run_calibration_experiment(parser.parse_args(argv).output)
    print(json.dumps({k: result[k] for k in ("clean_heldout", "adversarial_heldout",
                                           "sparse_calibration", "uniform_single_bad_row_budget")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
