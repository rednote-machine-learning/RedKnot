#!/usr/bin/env python3
"""Score the 300-question English LongBench RedKnot Full / S1.6 replay (standard library only).

python score_longbench300.py --data DATA --results NATIVE_RESULTS --out REPORT
Use --reference to rescore the original completed groups from the larger LongBench suite.
"""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path
import re
import statistics
import string
import tempfile
import zipfile

DATASETS = ("hotpotqa", "2wikimqa", "musique")
ARMS = ("Full", "S1.6")
GROUPS = tuple(f"lb-{n:04d}" for n in range(75))
CASE_IDS = tuple(i for start in (1000, 2000, 3000) for i in range(start, start + 100))
META_FIELDS = ("index", "dataset", "source_id", "source_index", "composition_mode", "length_band",
               "actual_document_count", "input_tokens", "input_ids_sha256", "question", "references")
META_DIGEST = "284e99a9f5bb68703f7e2f4c19d760a3f33837af1c3d8b9e514d14079768196a"
GROUP_DIGEST = "555f136453c1c2ef4707b547ed913bd0f65cee62da49993c0745d09368da808f"
REFERENCE_OUTPUT_DIGEST = "dd8077ada7152235f1ec175ba4d800f061e3d4264231113f99d7e7c1416d4582"
REFERENCE_CONTRACT = "bfe129cde7b6635a7089fab64e5860fa3e1e3be303965d4dd7faddc7ca8e85df"
METRIC_SOURCE_SHA256 = "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323"
PUNCTUATION = str.maketrans("", "", string.punctuation)
ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def sealed(path):
    value = read_json(path)
    signature = value.pop("sha256", None)
    require(digest(value) == signature, f"Invalid or missing result checksum at {path}")
    return value


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
    Path(tmp.name).replace(path)


def normalize(text):
    """Apply the English token normalization used by LongBench's QA F1 metric."""
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCTUATION)).split())


def qa_scores(prediction, references):
    """Compute the largest reference token F1, literal EM and normalized ordered EM."""
    normalized = normalize(prediction)
    pred = normalized.split()
    pred_counts = Counter(pred)
    max_f1 = 0
    norm_em = False
    for reference in references:
        gold = normalize(reference)
        norm_em |= (normalized == gold)
        gold_tokens = gold.split()
        matched = sum((pred_counts & Counter(gold_tokens)).values())
        if matched:
            precision = matched / len(pred)
            recall = matched / len(gold_tokens)
            max_f1 = max(max_f1, (2 * precision * recall) / (precision + recall))
    return max_f1, prediction in references, norm_em


def first_divergence(left, right):
    for position, (token, other) in enumerate(zip(left, right), 1):
        if token != other:
            return position
    return min(len(left), len(right)) + 1 if len(left) != len(right) else None


def measured_answer(meta, ids, prediction, raw_text, finish, ttft_ns, elapsed_ns):
    count = len(ids)
    require(count > 0 and all(type(t) is int and t >= 0 for t in ids), f"Invalid output IDs for {meta['index']}")
    require(type(ttft_ns) is int and type(elapsed_ns) is int and 0 < ttft_ns <= elapsed_ns
            and finish in ("stop", "length"), f"Invalid timing or finish reason for {meta['index']}")
    require(isinstance(prediction, str) and isinstance(raw_text, str), "Missing decoded output text")
    score, raw_em, norm_em = qa_scores(prediction, meta["references"])
    decode = (elapsed_ns - ttft_ns) / 1e9
    return dict(prediction_text=prediction, complete_raw_text=raw_text, output_ids=ids,
                output_ids_sha256=digest(ids), finish_reason=finish, output_tokens=count,
                score=score, raw_exact=raw_em, normalized_exact=norm_em, ttft_s=ttft_ns / 1e9,
                duration_s=elapsed_ns / 1e9, after_first_token_count=count - 1, decode_s=decode,
                decode_tokens_per_s=(count - 1) / decode if count > 1 and decode > 0 else None)


def arm_record(meta, request):
    count = len(request["output_ids"])
    require(request["output_tokens"] == count, f"Output token count mismatch for {meta['index']}")
    if request["finish_reason"] == "length":
        require(count == request["model_context_output_capacity"],
                f"Unexpected generation limit for {meta['index']}")
    received = request["received_ns"]
    return measured_answer(meta, request["output_ids"], request["prediction_text"], request["text"],
                           request["finish_reason"], request["first_token_ns"] - received,
                           request["completed_ns"] - received)


def read_arm(group, name, arm, contract, targets):
    raw = sealed(group / (arm + ".json"))
    sparse = arm == "S1.6"
    require(raw["status"] == "complete" and raw["actual_gpu_executed"] is True and
            raw["phase"] == "quality" and raw["group_id"] == name and raw["contract_id"] == contract and
            raw["method"] == arm and raw["native_method"] == ("Cone15" if sparse else "Full") and
            raw["input_indices"] == raw["target_indices"] == targets and raw["selector_strategy"] == "max" and
            raw["request_batch_size"] == raw["global_real_request_count"] == 4 and raw["ep_size"] == 8 and
            raw["quality_artificial_limit"] is None and raw["main_and_draft_graphs_enabled"] is True and
            raw["offline_preparation"]["offline_work_in_ttft"] is False and
            raw["gemm_revision"] == "s16-sm90-runtime-m-asm-ask-v1", f"Wrong geometry or method at {name}/{arm}")
    requests = raw["requests"]
    require([q["index"] for q in requests] == targets and [q["owner_rank"] for q in requests] == [0, 1, 2, 3],
            f"Wrong request order at {name}/{arm}")
    for q in requests:
        gemm, ctrl = q["gemm"], q["controller"]
        require(q["actual_active_request_count"] == 4 and gemm["enabled"] is sparse and
                gemm["method_uses_dynamic_gemm"] is sparse and gemm["gemm_revision"] == raw["gemm_revision"] and
                ctrl["input_ids_sha256"] == q["input_ids_sha256"] and ctrl["model_boundary_restored"] and
                ctrl["decode_native"], f"Invalid execution evidence for {name}/{arm}/{q['index']}")
        if sparse:
            require(ctrl["sparse_method"] == "Cone15" and ctrl["mandatory_tail128_complete"] and
                    0 < ctrl["selected_tokens"] <= q["input_tokens"], "Unexpected S1.6 selection")
    return requests


def collect(data, results, expected_contract, reference_mode):
    if reference_mode:
        contract = REFERENCE_CONTRACT
        wall = None
    else:
        terminal = sealed(results / "run.json")
        require(terminal["status"] == "complete" and (terminal["group_count"], terminal["target_count"]) == (75, 300),
                "Fresh replay must contain a completed 75-group, 300-question run")
        require({p.parent.name for p in (results/"groups").glob("*/group.json")} == set(GROUPS),
                "Missing or extra completed groups")
        contract = terminal["contract_id"]
        wall = terminal["finished"] - terminal["began"]
        require(wall >= 0, "Invalid total runner time")
    if expected_contract is not None:
        require(contract == expected_contract, "Contract does not match the requested run")
    metadata = {i: read_json(data/"index"/f"{i}.json") for i in CASE_IDS}
    require(digest([[metadata[i][k] for k in META_FIELDS] for i in CASE_IDS]) == META_DIGEST,
            "Inputs, references, source questions or dataset assignments differ from the frozen cohort")
    require(len({(r["dataset"], r["source_id"]) for r in metadata.values()}) == 300,
            "A source question is repeated")
    rows, groups = [], []
    for ordinal, name in enumerate(GROUPS, 1):
        plan = read_json(data/"groups"/f"{name}.json")
        targets = plan["target_indices"]
        require(plan["group_id"] == name and targets == plan["indices"] and len(set(targets)) == len(targets) == 4,
                f"Invalid frozen batch {name}")
        groups.append([name, targets])
        group = results/"groups"/name
        commit = sealed(group/"group.json")
        require(commit["status"] == "complete" and commit["methods"] == list(ARMS) and
                commit["group_id"] == name and commit["contract_id"] == contract and
                commit["indices"] == commit["target_indices"] == targets and commit["completed_targets"] == 4*ordinal,
                f"Invalid paired completion in {name}")
        left, right = (read_arm(group, name, arm, contract, targets) for arm in ARMS)
        for owner, (left_req, right_req) in enumerate(zip(left, right)):
            meta = metadata[targets[owner]]
            for req in (left_req, right_req):
                require(all(req[k] == meta[k] for k in ("index", "dataset", "source_id", "input_tokens", "input_ids_sha256")),
                        f"Mismatched prompt for case {meta['index']}/{name}")
            row = {k: meta[k] for k in META_FIELDS if k != "actual_document_count"}
            row.update(group=name, owner_rank=owner, batch_size=4, chunk_count=meta["actual_document_count"],
                       metric="qa_f1_score", composition_scope=meta.get("composition_scope"),
                       arms={a: arm_record(meta, q) for a, q in zip(ARMS, (left_req, right_req))},
                       first_divergence_token_1based=first_divergence(left_req["output_ids"], right_req["output_ids"]))
            a, b = (row["arms"][m] for m in ARMS)
            row["speedup_vs_full"] = a["ttft_s"] / b["ttft_s"]
            row["score_change_pp"] = 100 * (b["score"] - a["score"])
            rows.append(row)
    require(digest(groups) == GROUP_DIGEST, "B4 group composition or order changed")
    rows.sort(key=lambda r: r["index"])
    require([r["index"] for r in rows] == list(CASE_IDS), "Missing or duplicate paired targets")
    return rows, contract, wall


def summarize(rows):
    full, sparse = ([r["arms"][m] for r in rows] for m in ARMS)
    scores = [(a["score"], b["score"]) for a, b in zip(full, sparse)]
    out = dict(paired=len(rows), identical_outputs=sum(r["first_divergence_token_1based"] is None for r in rows),
               min_input_tokens=min(r["input_tokens"] for r in rows), max_input_tokens=max(r["input_tokens"] for r in rows),
               full_ge_0_8=sum(x >= .8 for x, y in scores),
               retained_ge_0_8=sum(x >= .8 and y >= .8 for x, y in scores),
               crossed_below_0_8=sum(x >= .8 and y < .8 for x, y in scores),
               severe_drop=sum(x >= .8 and y < .8 and x-y >= .5-1e-12 for x, y in scores),
               full_one_to_zero=sum(x == 1 and y == 0 for x, y in scores))
    for arm, items in zip(ARMS, (full, sparse)):
        measurable = [r for r in items if r["decode_tokens_per_s"] is not None]
        after_first = sum(r["after_first_token_count"] for r in measurable)
        decode = sum(r["decode_s"] for r in measurable)
        out[arm] = dict(score_100=round(100*sum(r["score"] for r in items)/len(items), 2),
                        raw_exact_count=sum(r["raw_exact"] for r in items),
                        normalized_exact_count=sum(r["normalized_exact"] for r in items),
                        mean_ttft_s=statistics.mean(r["ttft_s"] for r in items),
                        mean_total_s=statistics.mean(r["duration_s"] for r in items),
                        mean_output_tokens=statistics.mean(r["output_tokens"] for r in items),
                        total_output_tokens=sum(r["output_tokens"] for r in items),
                        decode_tokens_per_s=after_first/decode if decode > 0 else None,
                        decode_rate_measurable_requests=len(measurable),
                        finish_reasons=dict(Counter(r["finish_reason"] for r in items)))
    out["speedup_ratio_of_mean_ttft"] = out["Full"]["mean_ttft_s"]/out["S1.6"]["mean_ttft_s"]
    out["speedup_ratio_of_mean_total"] = out["Full"]["mean_total_s"]/out["S1.6"]["mean_total_s"]
    out["score_change_pp"] = 100*statistics.mean(y-x for x, y in scores)
    return out


def answer_html(dataset, rows, reference_mode):
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>' + escape(dataset) + '</title>',
            '<style>body{font:16px/1.45 system-ui;max-width:95rem;margin:2rem auto;padding:0 1rem}'
            'article{border-top:1px solid #aaa;margin:2rem 0}section{display:grid;gap:1em;'
            'grid-template-columns:repeat(auto-fit,minmax(min(100%,23rem),1fr))}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}</style>',
            '<p><a href="../index.html">Summary</a></p><h1>' + escape(dataset) + '</h1>']
    if reference_mode:
        page.append("<p>Historical GPU inference. Answers and latency samples come from the original run.</p>")
    for row in rows:
        page.extend([f'<article id="q{row["index"]}"><h2>#{row["index"]} · {row["input_tokens"]:,} input tokens</h2>',
                     f'<p>{row["chunk_count"]} chunks · {escape(row["composition_mode"])} · first output-token divergence '
                     + str(row["first_divergence_token_1based"] or "none") + '</p><h3>Question</h3><pre>'
                     + escape(row["question"]) + '</pre><h3>Reference answers</h3><ul>',
                     *('<li>'+escape(ref)+'</li>' for ref in row["references"]), '</ul><section>'])
        for arm in ARMS:
            a = row["arms"][arm]
            speed = "n/a" if a["decode_tokens_per_s"] is None else f'{a["decode_tokens_per_s"]:.2f}'
            page.extend([f'<div><h3>{arm}</h3><p>F1 {100*a["score"]:.2f} · normalized EM {int(a["normalized_exact"])}'
                         f' · TTFT {a["ttft_s"]:.3f} s · total {a["duration_s"]:.3f} s · {a["output_tokens"]} tokens'
                         f' · decode {speed} tokens/s</p>', '<pre>'+escape(a["prediction_text"])+'</pre></div>'])
        page.append('</section></article>')
    return "\n".join(page+['</html>'])


def index_html(result):
    header = ('<th>Subset</th><th>N</th><th>Full F1</th><th>S1.6 F1</th><th>Full TTFT (s)</th>'
              '<th>S1.6 TTFT (s)</th><th>TTFT speedup</th><th>Full ≥0.8 retained</th>')
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>LongBench 300</title>',
            '<style>body{font:16px/1.5 system-ui;max-width:80rem;margin:2rem auto;padding:0 1rem}'
            'table{border-collapse:collapse}th,td{text-align:right;border-bottom:1px solid #bbb;padding:.4rem}'
            'th:first-child,td:first-child{text-align:left}</style><h1>LongBench 300 · Full / S1.6</h1>',
            ('<p>Historical GPU inference. The recorded answers are rescored locally; timings come from the original run.</p>'
             if result["mode"] == "rescore_historical_outputs" else '<p>Fresh GPU replay.</p>'),
            '<p>'+escape(result["measurement"])+'</p><p>'+escape(result["decode_rate_definition"])+'</p>',
            '<p><a href="summary.json">Complete JSON statistics</a></p>']
    for title, groups in (("Dataset", {"All":result["overall"], **result["by_dataset"]}),
                          ("Input length", result["by_configured_length_band"]),
                          ("Prompt composition", result["by_mode"])):
        page.extend(['<h2>'+title+'</h2><table><thead><tr>'+header+'</tr></thead><tbody>'])
        for label, s in groups.items():
            name = '<a href="answers/'+label+'.html">'+label+'</a>' if label in DATASETS else escape(label)
            a, b = (s[m] for m in ARMS)
            page.append(f'<tr><td>{name}</td><td>{s["paired"]}</td><td>{a["score_100"]:.2f}</td><td>{b["score_100"]:.2f}</td>'
                        f'<td>{a["mean_ttft_s"]:.3f}</td><td>{b["mean_ttft_s"]:.3f}</td>'
                        f'<td>{s["speedup_ratio_of_mean_ttft"]:.3f}×</td><td>{s["retained_ge_0_8"]}/{s["full_ge_0_8"]}</td></tr>')
        page.append('</tbody></table>')
    return "\n".join(page + ['</html>'])


def score_cohort(data, results, report, expected_contract=None, reference_mode=False):
    """Write summary.json/index.html and complete, paired JSONL/HTML by dataset."""
    data, results, report = map(Path, (data, results, report))
    rows, contract, wall = collect(data, results, expected_contract, reference_mode)
    return write_report(rows, report, contract, wall, reference_mode)


def write_report(rows, report, contract, wall, reference_mode):
    output_digest = digest([[r["index"], r["input_ids_sha256"], *(r["arms"][m]["output_ids_sha256"] for m in ARMS)] for r in rows])
    if reference_mode:
        require(output_digest == REFERENCE_OUTPUT_DIGEST, "Reference outputs differ from their published checksum")
    by_dataset = {d: [r for r in rows if r["dataset"] == d] for d in DATASETS}
    def split(sample, key):
        labels = {r[key] for r in sample}
        order = sorted(labels, key=lambda v: int(v[:-1])) if key == "length_band" else sorted(labels)
        return {label:summarize([r for r in sample if r[key] == label]) for label in order}
    rollup = dict(schema="redknot-longbench300-report-v1", status="complete", expected=300, completed=300, group_count=75,
                  mode="rescore_historical_outputs" if reference_mode else "score_gpu_replay", contract_id=contract,
                  metric_name="LongBench v1 English QA token F1 (maximum across references)",
                  upstream_longbench_metrics_sha256=METRIC_SOURCE_SHA256, output_ids_digest=output_digest,
                  historical_output_ids_digest=REFERENCE_OUTPUT_DIGEST, all_output_ids_match_historical=output_digest==REFERENCE_OUTPUT_DIGEST,
                  measurement="Online TTFT is measured from request receipt to the first raw output token. The model is loaded, and document KV banks are prepared separately. Speedup is the ratio of mean TTFTs.",
                  decode_rate_definition="Per-request decode rate uses output token IDs after the first, including a recorded stop token, divided by time from first token to terminal. Aggregate rate divides the summed count by the summed decode time for multi-token requests with a positive measured interval. Overlapping requests accumulate independent generation time; server batch throughput uses a different clock.",
                  runner_wall_s=wall, overall=summarize(rows), by_dataset={d:summarize(rs) for d,rs in by_dataset.items()},
                  by_configured_length_band=split(rows,"length_band"), by_mode=split(rows,"composition_mode"),
                  by_dataset_and_mode={d:split(rs,"composition_mode") for d,rs in by_dataset.items()},
                  by_dataset_and_length={d:split(rs,"length_band") for d,rs in by_dataset.items()})
    for dataset, sample in by_dataset.items():
        write(report/"answers"/(dataset+".jsonl"), "".join(canonical(r)+"\n" for r in sample))
        write(report/"answers"/(dataset+".html"), answer_html(dataset, sample, reference_mode))
    write(report/"index.html", index_html(rollup))
    write(report/"summary.json", json.dumps(rollup, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return rollup


def render_reference(data, archive, output):
    """Validate and rescore compact historical GPU outputs against the frozen local dataset."""
    data, archive, output = map(Path, (data, archive, output))
    with zipfile.ZipFile(archive) as stream:
        names = stream.infolist()
        require(len(names) == 1 and names[0].filename == "answers.jsonl" and names[0].file_size == 145127,
                "Unexpected reference archive contents")
        payload = stream.read(names[0])
    require(hashlib.sha256(payload).hexdigest() == "97d27ba883badef5a9f7606ae7ef647437685ab9bc443d2d391b4e986bf41624",
            "Historical token, text or timing records differ from the frozen run")
    meta = {i: read_json(data/"index"/f"{i}.json") for i in CASE_IDS}
    require(digest([[meta[i][k] for k in META_FIELDS] for i in CASE_IDS]) == META_DIGEST,
            "Metadata, references or prompt token hashes differ from the frozen 300 questions")
    plans = []
    assignment = {}
    for group in GROUPS:
        plan = read_json(data/"groups"/f"{group}.json")
        ids = plan["indices"]
        require(plan["group_id"] == group and plan["target_indices"] == ids and len(ids) == 4,
                f"Malformed B4 group {group}")
        plans.append([group, ids])
        for owner, index in enumerate(ids):
            assignment[index] = (group, owner)
    require(digest(plans) == GROUP_DIGEST and set(assignment) == set(CASE_IDS), "Group assignments changed")
    records = [json.loads(line) for line in payload.splitlines()]
    require([r["index"] for r in records] == list(CASE_IDS), "Missing or repeated recorded answers")
    rows = []
    for item in records:
        i = item["index"]
        m = meta[i]
        require(item["input_ids_sha256"] == m["input_ids_sha256"], f"Historical answer prompt mismatch for {i}")
        group, owner = assignment[i]
        row = {k: m[k] for k in META_FIELDS if k != "actual_document_count"}
        values = {}
        for arm in ARMS:
            a = item["arms"][arm]
            values[arm] = measured_answer(m, a["ids"], a["prediction"], a["raw"], a["finish"], a["ttft_ns"], a["elapsed_ns"])
        f, s = (values[k] for k in ARMS)
        row.update(group=group, owner_rank=owner, batch_size=4, chunk_count=m["actual_document_count"],
                   metric="qa_f1_score", composition_scope=m.get("composition_scope"), arms=values,
                   first_divergence_token_1based=first_divergence(f["output_ids"], s["output_ids"]),
                   speedup_vs_full=f["ttft_s"]/s["ttft_s"], score_change_pp=100*(s["score"]-f["score"]))
        rows.append(row)
    return write_report(rows, output, REFERENCE_CONTRACT, None, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--results", type=Path, help="Directory of native paired B4 results")
    where.add_argument("--reference-archive", type=Path, help="Rescore the compact historical_reference.zip on CPU")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-contract", help="Verify the new run's native contract ID")
    parser.add_argument("--reference", action="store_true", help="Score only the frozen 75 groups of the published benchmark")
    args = parser.parse_args()
    try:
        if args.reference_archive:
            if args.expected_contract and args.expected_contract != REFERENCE_CONTRACT:
                parser.error("The archived historical contract differs from --expected-contract")
            result = render_reference(args.data, args.reference_archive, args.out)
        else:
            result = score_cohort(args.data, args.results, args.out, args.expected_contract, args.reference)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    for d, r in result["by_dataset"].items():
        print(f'{d}: n={r["paired"]} Full/S1.6 F1={r["Full"]["score_100"]:.2f}/{r["S1.6"]["score_100"]:.2f} '
              f'TTFT speedup={r["speedup_ratio_of_mean_ttft"]:.3f}x')
    print(args.out/"index.html")


if __name__ == "__main__":
    main()
