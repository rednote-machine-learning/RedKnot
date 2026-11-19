#!/usr/bin/env python3
"""Reproduce Full / RedKnot on 21 frozen LongBench datasets (100 targets each)."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

from longbench21.dataset import prepare_data, verify_dataset
from longbench21.report import load_jsonl, render_report
from longbench21.scoring import checked, group_scores, load_evaluator, per_question

ASSETS = Path(__file__).resolve().parent / "longbench21"
DEFAULT_OUTPUT = ASSETS.parents[1] / "benchmark-output" / "redknot-longbench21"
RECORDED = ASSETS / "recorded"


def inventory(data):
    manifest = json.loads((data / "complete.json").read_bytes())
    if (manifest["target_count"], manifest["group_count"]) != (2100, 527):
        raise ValueError("Expected the frozen 2,100 targets in 527 B4 groups")


def paired_rows(data, results, cache, evaluator, *, require_complete):
    revision = results / "gemm-revision.json"
    if not revision.is_file() and not require_complete and not (results / "groups").exists():
        return []
    meta, _ = checked(revision)
    if Path(meta["manifest"]).resolve() != data.resolve():
        raise ValueError("The results belong to a different input cache")
    rows = []
    for commit in sorted((results / "groups").glob("lb-*/group.json")):
        group = commit.parent
        saved, _ = checked(commit)
        plan = json.loads((data / "groups" / (group.name + ".json")).read_bytes())
        if (saved["contract_id"] != meta["contract_id"] or saved["indices"] != plan["indices"]
                or saved["target_indices"] != plan["target_indices"]):
            raise ValueError(f"Committed results do not match the frozen group {group.name}")
        rows.extend(group_scores(group, data, evaluator, cache))
    if require_complete:
        terminal, _ = checked(results / "run.json")
        if (terminal["contract_id"], terminal["status"], terminal["target_count"], terminal["group_count"]) != (
                meta["contract_id"], "complete", 2100, 527) or len(rows) != 2100:
            raise ValueError("Native run did not complete the 2,100 paired targets")
    return rows


def reference_rows(evaluator):
    manifest = json.loads((RECORDED / "provenance.json").read_bytes())
    path = RECORDED / "answers.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]["sha256"]:
        raise ValueError("Archived paired log differs from its recorded checksum")

    def score(row, text):
        return per_question(evaluator, row["dataset"], text, row["references"], row.get("all_classes", []))

    rows = load_jsonl(path, score_fn=score)
    original_targets = {1000 * ds + i for ds in range(1, 6) for i in range(100)}
    if {row["index"] for row in rows} != original_targets:
        raise ValueError("Expected exactly 100 historical targets from each of the first five datasets")
    return rows


def print_report(summary, report):
    print(f"\n{summary['completed']:,} / {summary['expected']:,} paired targets")
    print(f"{'Dataset':<22} {'N':>4} {'Full score':>12} {'RedKnot score':>14} "
          f"{'Full TTFT/s':>12} {'RedKnot TTFT/s':>16} {'Speedup':>9}")
    for dataset, item in summary["by_dataset"].items():
        if item["paired"]:
            full, rk = item["Full"], item["RedKnot"]
            print(f"{dataset:<22} {item['paired']:>4} {full['score_100']:>12.2f} {rk['score_100']:>14.2f} "
                  f"{full['mean_ttft_s']:>12.2f} {rk['mean_ttft_s']:>16.2f} "
                  f"{item['speedup_ratio_of_mean_ttft']:>8.2f}x")
    print(f"Report: {report / 'index.html'}\nAnswers: {report / 'answers.jsonl'}\nSummary: {report / 'summary.json'}")


def output_dir(path, mode, reopen):
    if reopen:
        if path is None or not (path.expanduser() / "benchmark.json").is_file():
            raise ValueError("Use --output to specify an existing GPU run directory")
        return path.expanduser().resolve()
    if path is not None:
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError("Choose a new output directory, or use --resume / --report-only on a prior run")
        return path
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(mkdtemp(prefix=f"{mode}-{stamp}-", dir=DEFAULT_OUTPUT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="verify all local inputs, chunk spans and batches on CPU")
    modes.add_argument("--check-runtime", action="store_true", help="validate the checkpoint and all native B4 inputs on CPU")
    modes.add_argument("--reference", action="store_true", help="rescore the 500 archived pairs and rebuild their report on CPU")
    modes.add_argument("--report-only", action="store_true", help="score committed pairs in an existing run, including a partial run")
    modes.add_argument("--resume", action="store_true", help="continue an interrupted run at --output with the same --cache")
    parser.add_argument("--output", type=Path, help="run directory; defaults to a new directory under benchmark-output/")
    parser.add_argument("--cache", type=Path, help="persistent local input and evaluator cache")
    parser.add_argument("--model-path", type=Path, default=os.environ.get("REDKNOT_MODEL_PATH"), help="local DeepSeek-V4.1-Flash-1017 checkpoint")
    parser.add_argument("--runtime-root", type=Path, default=os.environ.get("REDKNOT_RUNTIME_ROOT"), help="optional external runtime in place of the bundled implementation")
    args = parser.parse_args()
    cache = (args.cache or DEFAULT_OUTPUT / "cache").expanduser().resolve()
    if not (args.reference or args.check or args.report_only or args.resume or args.model_path):
        parser.error("Set --model-path or REDKNOT_MODEL_PATH for the GPU runner or its CPU preflight")
    if args.check:
        print(json.dumps(verify_dataset(prepare_data(ASSETS, cache)), ensure_ascii=False, indent=2))
        return

    run = output_dir(args.output, "reference" if args.reference else "run", args.resume or args.report_only)
    report = run / "report"
    print(f"Run directory: {run}", flush=True)
    if args.reference:
        evaluator = load_evaluator(cache / "official-scoring")
        summary = render_report(reference_rows(evaluator), report, mode="historical_reference")
        print_report(summary, report)
        return
    # Persist the run identity before materializing data or compiling extensions.
    # A preflight interruption can then be resumed just like a partially measured B4.
    marker = run / "benchmark.json"
    schema = "redknot-longbench21-b4-v1"
    if args.resume or args.report_only:
        binding = json.loads(marker.read_bytes())
        if binding.get("schema") != schema:
            raise ValueError("The selected directory belongs to a different benchmark")
        if args.cache is not None and str(cache) != binding["cache"]:
            raise ValueError("Resume with the same input cache used by this run")
        cache = Path(binding["cache"])
        for attr in ("model_path", "runtime_root"):
            original = binding[attr]
            requested = getattr(args, attr)
            if not args.report_only and requested is not None and str(requested.expanduser().resolve()) != original:
                raise ValueError(f"Run identity changed: {attr}")
            setattr(args, attr, Path(original) if original is not None else None)
    else:
        binding = dict(schema=schema, cache=str(cache),
                       model_path=str(args.model_path.expanduser().resolve()),
                       runtime_root=str(args.runtime_root.expanduser().resolve()) if args.runtime_root else None)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(binding, stream, ensure_ascii=False)
    data = prepare_data(ASSETS, cache)
    inventory(data)
    results = run / "results"
    should_run = not args.report_only and not (args.resume and (results / "run.json").is_file())
    if should_run:
        from longbench300.local_backend import run_local
        print("Checking the 527 original B4 groups and checkpoint …", flush=True)
        result = run_local(args.runtime_root, args.model_path, data, results, args.check_runtime, target_count=2100)
        if args.check_runtime:
            print(json.dumps(result, indent=2))
            return

    # Independent of GPU inference; a live snapshot reads only atomically committed pairs.
    with (run / ".report.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluator = load_evaluator(cache / "official-scoring")
        rows = paired_rows(data, results, run / ".score-cache", evaluator, require_complete=not args.report_only)
        summary = render_report(rows, report, mode="inference")
    print_report(summary, report)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"Benchmark failed: {exc}")
