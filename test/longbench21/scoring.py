"""Score paired native RedKnot S1.6 and Full outputs using the pinned LongBench v1 evaluator."""
import ast
import difflib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ARMS = ("Full", "S1.6")
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
EVAL_SHA = {
    "eval.py": "1a3acfc25d9b053e9bb75c479f7e385d0cb9989f0f7115346b7d632655967721",
    "metrics.py": "e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323",
}
WHEELS = {
    "fuzzywuzzy-0.18.0-py2.py3-none-any.whl": "928244b28db720d1e0ee7587acf660ea49d7e4c632569cad4f1cd7e68a5f0993",
    "jieba-0.42.1-py3-none-any.whl": "745f04589953aba6e3eb4c94bb086e155ff047aa58a76bbc5e6eb7294bb87feb",
    "rouge-1.0.1-py3-none-any.whl": "28d118536e8c774dc47d1d15ec266479b4dd0914c4672ce117d4002789bdc644",
    "six-1.17.0-py2.py3-none-any.whl": "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def checked(path, *, require_seal=False):
    raw = path.read_bytes()
    obj = json.loads(raw)
    if require_seal:
        require("sha256" in obj, f"Missing record seal {path}")
    if "sha256" in obj:
        require(obj["sha256"] == digest(canonical({k: v for k, v in obj.items() if k != "sha256"})),
                f"Invalid record seal {path}")
    return obj, digest(raw)


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", delete=False) as stream:
        stream.write(contents)
        temp = Path(stream.name)
    os.replace(temp, path)


def write_json(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False).encode())


def load_evaluator(cache):
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "vendor.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        site = cache / ("pydeps-" + digest(canonical(WHEELS))[:20])
        if not (site / "ready").is_file():
            site.mkdir(exist_ok=True)
            for name, sha in WHEELS.items():
                whl = HERE / "vendor" / name
                require(digest(whl.read_bytes()) == sha, f"Changed dependency {name}")
                with zipfile.ZipFile(whl) as src:
                    src.extractall(site)
            write(site / "ready", canonical(WHEELS))
    sys.path.insert(0, str(site))
    # The official requirements specify fuzzywuzzy without its optional Levenshtein backend.
    from fuzzywuzzy import fuzz
    import jieba
    import rouge
    import six
    for module in (fuzz, jieba, rouge, six):
        require(site in Path(module.__file__).resolve().parents, f"External scoring dependency {module.__file__}")
    fuzz.SequenceMatcher = difflib.SequenceMatcher
    jieba.dt.tmp_dir = str(cache)
    for name, sha in EVAL_SHA.items():
        require(digest((HERE / "official" / name).read_bytes()) == sha, f"Changed official source {name}")
    spec = importlib.util.spec_from_file_location("redknot_longbench_metrics", HERE / "official" / "metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    # Only LongBench-E's length statistics need NumPy. Read the pinned standard
    # evaluator's metric registry and invoke its original metric functions.
    tree = ast.parse((HERE / "official" / "eval.py").read_bytes())
    registry = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(lhs, ast.Name) and lhs.id=="dataset2metric" for lhs in node.targets))
    return SimpleNamespace(dataset2metric={key.value: getattr(metrics, ref.id)
                                           for key, ref in zip(registry.keys, registry.values)})


def per_question(evaluator, dataset, text, references, classes):
    # Official eval.py::scorer rounds only after evaluating the entire dataset.
    # Keep the per-reference metric outputs unrounded so means use the same arithmetic.
    require(references and all(isinstance(v, str) for v in references), "Invalid reference answers")
    if dataset in FIRST_LINE:
        text = text.lstrip("\n").split("\n")[0]
    metric = evaluator.dataset2metric[dataset]
    return max(metric(text, ref, all_classes=classes) for ref in references)


def divergence(a, b):
    for pos, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return pos
    return min(len(a), len(b)) + 1 if len(a) != len(b) else None


def select_requests(group, arm, commit):
    value, sha = checked(group / (arm + ".json"), require_seal=True)
    require(value["status"]=="complete" and value["actual_gpu_executed"] and
            value["native_method"]==("Full" if arm=="Full" else "Cone15") and
            value["main_and_draft_graphs_enabled"] and value["quality_artificial_limit"] is None and
            value["method"] == arm and value["phase"] == "quality"
            and value["request_batch_size"] == 4 and value["ep_size"] == 8,
            f"Unexpected quality arm or GPU geometry {group}/{arm}")
    requests = value["requests"]
    require(value["contract_id"]==commit["contract_id"] and value["group_id"]==commit["group_id"]==group.name
            and value["target_indices"]==commit["target_indices"]
            and value["input_indices"]==commit["indices"]==[r["index"] for r in requests]
            and set(commit["target_indices"])<=set(commit["indices"]),"Raw B4 result and committed group disagree")
    require(len(requests) == value["request_batch_size"] and len({r["index"] for r in requests}) == len(requests)
            and len({r["owner_rank"] for r in requests}) == len(requests), f"Duplicate or missing requests in {group}/{arm}")
    for r in requests:
        require(r["received_ns"] < r["first_token_ns"] <= r["completed_ns"] and r["output_ids"]
                and isinstance(r["prediction_text"], str) and isinstance(r["text"], str),
                f"Missing completion evidence for {r['index']}/{arm}")
    return requests, sha


def group_scores(group, data, evaluator, cache):
    commit, group_sha=checked(group/"group.json", require_seal=True)
    require(commit["status"]=="complete" and tuple(commit["methods"])==ARMS, "Pair not complete")
    left, left_sha = select_requests(group, "Full", commit)
    right, right_sha = select_requests(group, "S1.6", commit)
    require([x["index"] for x in left] == [x["index"] for x in right], f"Different request order in {group}")
    indices={i:checked(data/"index"/f"{i}.json") for i in commit["target_indices"]}
    binding=digest(canonical([digest(Path(__file__).read_bytes()),EVAL_SHA,WHEELS,group_sha,left_sha,right_sha,
                              [row_sha for row,row_sha in indices.values()]]))
    saved=cache/(group.name+".json")
    if saved.exists():
        old,_=checked(saved, require_seal=True)
        if old["binding"]==binding:
            return old["cases"]
    result = []
    for full, candidate in zip(left, right):
        index = full["index"]
        if index not in indices:
            continue
        row, row_sha = indices[index]
        require(row["index"] == index and row["input_ids_sha256"] == full["input_ids_sha256"] == candidate["input_ids_sha256"]
                and row["input_tokens"] == full["input_tokens"] == candidate["input_tokens"]
                and full["owner_rank"] == candidate["owner_rank"], f"Mismatched input or owner {index}")
        ds = row["dataset"]
        refs = row["references"]
        classes = row["all_classes"]
        if ds in ("trec", "lsht"):
            require(isinstance(classes, list) and set(refs) <= set(classes), f"Missing official classes for {index}")
        require(row["composition_mode"] in ("native", "extended") and 5 <= row["actual_document_count"] <= 15,
                f"Unknown composition for {index}")
        arms = {}
        for arm, r in zip(ARMS, (full, candidate)):
            ids = r["output_ids"]
            pred = r["prediction_text"]
            arms[arm] = dict(prediction_text=pred, complete_raw_text=r["text"], output_ids_sha256=digest(canonical(ids)),
                             output_tokens=len(ids), finish_reason=r["finish_reason"],
                             score=per_question(evaluator, ds, pred, refs, classes),
                             ttft_s=(r["first_token_ns"] - r["received_ns"])/1e9,
                             duration_s=(r["completed_ns"] - r["received_ns"])/1e9)
        result.append(dict(index=index, dataset=ds, source_id=row["source_id"], source_index=row["source_index"],
                           question=row["question"], references=refs, all_classes=classes, composition_mode=row["composition_mode"],
                           composition_scope=row.get("composition_scope"), length_band=row["length_band"],
                           chunk_count=row["actual_document_count"], input_tokens=row["input_tokens"],
                           input_ids_sha256=row["input_ids_sha256"], manifest_file_sha256=row_sha,
                           group=group.name, batch_size=len(left), owner_rank=full["owner_rank"],
                           metric=evaluator.dataset2metric[ds].__name__, arms=arms,
                           first_divergence_token_1based=divergence(full["output_ids"], candidate["output_ids"]),
                           speedup_vs_full=arms["Full"]["ttft_s"]/arms["S1.6"]["ttft_s"],
                           score_change_pp=100 * (arms["S1.6"]["score"] - arms["Full"]["score"]),
                           native_files={k: v for k, v in zip(ARMS, (left_sha, right_sha))}))
    body=dict(binding=binding,cases=result)
    write_json(saved,dict(body,sha256=digest(canonical(body))))
    return result
