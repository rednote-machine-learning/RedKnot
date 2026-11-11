"""Independent CPU-only audit of sealed tail128-link full-model pilot evidence.

Only closed three-arm quality + three C1 repetitions enter scores. This module
does not import the controller, runner, Torch, native code or any GPU runtime.
It independently reconstructs tail selection and causal memory from absolute
positions and actual native layouts. It hashes every forty-MiB offline bank.
Producer selected-row totals and median summaries are not trusted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kv_cache_engram.core import digest, require, validate_digest
from gpu_pilot.tail_link_protocol import ARMS, CAPTURE_LAYERS, FFN_LAYERS, tail_link_protocol

SCHEMA = "tail-link-independent-readout-v1"
PHASE_SCHEMA = "kv-cache-engram-tail-link-phase-v1"
RUN_SCHEMA = "kv-cache-engram-tail-link-pilot-run-v1"
CASE_INDICES = (2, 3, 4, 5)
FRACTIONS = dict.fromkeys(ARMS)
MAX_JSON_BYTES = 8*1024**3


def frozen_sources(preflight):
    """Recheck every bound Python/dependency/native source, without importing it."""
    from gpu_pilot.tail_link_protocol import DEPENDENCIES
    require(DEPENDENCIES, "new reader dependency pins have not been frozen")
    stage = Path(__file__).resolve().parents[1]
    identity = preflight["identity"]
    root = Path(identity["experiment_root"])
    native = Path(identity["isolated_native_snapshot"])
    require(root.is_absolute() and native.is_absolute() and native.resolve() == stage.parent/"NativeSnapshot",
            "source roots differ from independent frozen deployment")
    recorded = identity.get("pilot_source_sha256")
    require(type(recorded) is dict and all(recorded.get(k) == v for k,v in DEPENDENCIES.items()),
            "preflight dependency pins differ from reader")
    paths = {"GatedEngram/"+str(p.relative_to(stage)):p for p in stage.rglob("*.py")
             if "__pycache__" not in p.parts}
    expected_stage = {k for k in recorded if k.startswith("GatedEngram/")}
    require(set(paths) == expected_stage, "frozen stage Python inventory changed")
    for name, expected in recorded.items():
        validate_digest(expected, "frozen source SHA")
        path = paths[name] if name in paths else root/name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "frozen source changed: "+name)
    native_files = identity["runtime_source_files_sha256"]
    require(digest(native_files) == identity["runtime_source_tree_sha256"], "native source aggregate differs")
    require(native.is_dir() and not native.is_symlink(), "native snapshot must be a real directory")
    git = ["git", "-c", "safe.directory="+str(native.resolve()), "-C", str(native)]
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], env=git_env).decode().strip()
    diff = subprocess.check_output([*git, "diff", "--no-ext-diff", "HEAD"], env=git_env)
    names = subprocess.check_output([*git, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], env=git_env)
    current_names = set()
    for raw in set(names.split(b"\0")):
        if raw:
            name = os.fsdecode(raw)
            path = native/name
            if path.is_file() and path.suffix in (".py", ".pyi", ".cpp", ".hpp", ".cu", ".h", ".toml"):
                require(not path.is_symlink(), "native runtime source symlink rejected")
                current_names.add(name)
    require(current_names == set(native_files) and head == identity["commit"]
            and hashlib.sha256(diff).hexdigest() == identity["source_diff_sha256"]
            and bool(diff) is identity["source_is_modified"], "native source inventory/commit/diff changed")
    for name, expected in native_files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "native source path escapes snapshot")
        path = native/name
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == expected, "private native source changed: "+name)
    return dict(pilot_source_sha256=digest(recorded), native_source_sha256=digest(native_files))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def sealed(value):
    require(type(value) is dict and value.get("sha256") == digest(
        {key: item for key, item in value.items() if key != "sha256"}), "canonical seal mismatch")
    return value


def read(path, evidence):
    path = Path(path)
    def identity(info):
        return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_JSON_BYTES,
            "missing, symlink, nonregular or oversized evidence: "+path.name)
    fd = os.open(path, os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
    with os.fdopen(fd,"rb") as handle:
        opened = os.fstat(handle.fileno())
        require(stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before), "evidence changed while opening")
        payload = handle.read(before.st_size+1)
        after = os.fstat(handle.fileno())
    require(len(payload) == before.st_size and identity(after) == identity(before)
            and identity(path.lstat()) == identity(before), "evidence changed during bounded read")
    def reject_constant(value):
        raise ValueError("non-finite JSON constant: "+value)
    value = sealed(json.loads(payload, object_pairs_hook=_unique, parse_constant=reject_constant))
    evidence[path.name] = dict(file_sha256=hashlib.sha256(payload).hexdigest(), seal_sha256=value["sha256"])
    return value


def verify_reference(reference, path, value, evidence, *, extras=()):
    expected = {"file", "file_sha256", "seal_sha256"}|set(extras)
    require(type(reference) is dict and set(reference) == expected
            and reference.get("file") == path.name and Path(reference["file"]).name == reference["file"]
            and reference.get("file_sha256") == evidence[path.name]["file_sha256"]
            and reference.get("seal_sha256") == value["sha256"],
            "closure reference does not bind original raw bytes and seal")


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), "invalid "+name)
    return value


def audit_bank_payload(path, expected_sha256):
    """Read a single 128x512 BF16 tail once; check SHA, finite values and stability."""
    path = Path(path)
    validate_digest(expected_sha256, "tail payload SHA")
    size = 128*512*2
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size == size, "tail payload missing or wrong size")
    fd = os.open(path, os.O_RDONLY|getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read(size+1)
        after = os.fstat(handle.fileno())
    require(len(raw) == size and ident(before) == ident(opened) == ident(after) == ident(path.lstat()),
            "tail payload changed during read")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "offline tail KV payload bytes changed")
    import array
    words = array.array("H", raw)
    if sys.byteorder != "little":
        words.byteswap()
    require(all((word & 0x7f80) != 0x7f80 for word in words), "nonfinite canonical BF16 tail payload")
    return dict(file_sha256=expected_sha256, bytes=size, finite_bfloat16=True)


def score(text, references):
    require(type(text) is str and type(references) is list and bool(references)
            and all(type(item) is str for item in references), "fixed answer/gold required")
    def norm(value):
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", value.lower())).split())
    words, outcomes = norm(text).split(), []
    for answer in references:
        gold = norm(answer).split()
        common = sum((Counter(words) & Counter(gold)).values())
        outcomes.append((float(norm(text) == norm(answer)),
                         2*common/(len(words)+len(gold)) if words or gold else 1.))
    return dict(answer_em=max(v[0] for v in outcomes), answer_token_f1=max(v[1] for v in outcomes))


def documents(row):
    docs = row.get("documents")
    require(type(docs) is list and len(docs) == 8, "independent audit requires eight raw document bindings")
    previous, keys, spans = None, set(), []
    for doc in docs:
        lo, hi, key = doc.get("input_start"), doc.get("input_end"), doc.get("token_ids_sha256")
        validate_digest(key, "document token SHA")
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= row["input_tokens"]
                and hi-lo == 32768 and (previous is None or lo == previous)
                and key not in keys, "document support, distinctness or length mismatch")
        previous = hi
        keys.add(key)
        spans.append((lo, hi, key))
    return spans


def mask_reference(positions, start, end, spans):
    """Independent absolute-position policy; no frequency or learned selector."""
    _positions(positions, start, end)
    selected, non_document, document_tail = [], [], []
    for i, p in enumerate(positions):
        doc = next(((lo, hi, key) for lo, hi, key in spans if lo <= p < hi), None)
        if doc is None:
            selected.append(i); non_document.append(i)
        elif p >= doc[1]-128:
            selected.append(i); document_tail.append(i)
    return selected, non_document, document_tail


def sparse_identity(reference, candidate, row, qualification_sha256):
    """Strict same-policy sparse parity; never an equivalence-to-full claim."""
    validate_digest(qualification_sha256,"native all-selected qualification")
    arm = reference.get("arm")
    require(arm in ARMS[1:] and candidate.get("arm") == arm, "same-policy qualification arm mismatch")
    for phase,mode in ((reference,"reference_sparse"),(candidate,"sensitive")):
        sealed(phase)
        require(phase.get("schema")==PHASE_SCHEMA and phase.get("mode")==mode
                and phase.get("dynamic_document_swa") is (arm == "tail128_dynamic")
                and phase.get("arm")==arm and phase.get("fraction") is None
                and phase.get("quality_eligible") is False and phase.get("timing_valid") is False
                and phase.get("index")==row["index"]
                and phase.get("input_ids_sha256")==row["input_ids_sha256"],
                "sparse reference/candidate phase identity mismatch")
        record=phase["record"]; control=phase["controller"]
        require(control.get("same_policy_causal_oracle_checked") is True,
                "same-policy qualification omitted tail mapping/restore oracle")
        audit_controller(control,row,phase,documents(row))
        require(record.get("success") is True and record.get("rank")==0
                and record.get("input_tokens")==row["input_tokens"]
                and record.get("input_ids_sha256")==row["input_ids_sha256"]
                and len(record.get("output_ids",[]))==1 and record.get("natural_EOS") is False
                and record.get("output_ids_sha256")==digest(record["output_ids"]),
                "sparse qualification output/input binding mismatch")
        fingerprints=control.get("sparse_identity_fingerprints")
        chunks=[(s,min(s+8192,row["input_tokens"])) for s in range(0,row["input_tokens"],8192)]
        expected={(s,e,l) for s,e in chunks for l in range(40)}
        require(type(fingerprints) is list and len(fingerprints)==len(expected)
                and control.get("sparse_identity_check_count")==len(expected)
                and control.get("sparse_identity_fingerprints_sha256")==digest(fingerprints),
                "full sparse layer/chunk fingerprints required")
        seen=set()
        for item in fingerprints:
            key=tuple(item.get("key",[])); value=item.get("value",{})
            require(key in expected and key not in seen
                    and set(value)=={"output","lazy"}
                    and set(value["lazy"])=={"residual","post_layer_mix","comb_res_mix","pre_layer_mix"},
                    "sparse output/all4 mHC field coverage differs")
            seen.add(key)
            require(value["output"] is not None,"sparse decoder output absent")
            _fingerprint(value["output"])
            for v in value["lazy"].values(): _fingerprint(v)
    for name in ("manifest_sha256","execution_configuration_sha256","tail_link_protocol_sha256",
                 "empty_ep_qualification_sha256"):
        require(reference.get(name)==candidate.get(name),"sparse qualification identity changed")
    require(candidate.get("verify_sparse") is True
            and candidate["controller"].get("sparse_identity_exact") is True
            and reference["controller"]["sparse_identity_fingerprints"]
                ==candidate["controller"]["sparse_identity_fingerprints"]
            and reference["record"]["output_ids"]==candidate["record"]["output_ids"],
            "sparse same-policy reference byte parity failed")
    fields=("start","end","layer","current_positions","selected_local_rows")
    require([{k:m[k] for k in fields} for m in reference["controller"]["selection"]]
            ==[{k:m[k] for k in fields} for m in candidate["controller"]["selection"]],
            "sparse reference and production masks differ")
    body=dict(schema="tail-link-sparse-qualification-v1",passed=True,arm=arm,
        index=row["index"],input_tokens=row["input_tokens"],input_ids_sha256=row["input_ids_sha256"],
        qualification_sha256=qualification_sha256,reference_sha256=reference["sha256"],
        candidate_sha256=candidate["sha256"],
        identity_check_count=candidate["controller"]["sparse_identity_check_count"],
        identity_fingerprints_sha256=candidate["controller"]["sparse_identity_fingerprints_sha256"],
        quality_eligible=False,timing_valid=False,equivalence_to_full_model_claim=False,
        exact_same_policy_reference=True)
    return dict(body,sha256=digest(body))


def _fingerprint(value):
    if value is None:
        return
    require(type(value) is dict and set(value) == {"shape", "dtype", "bytes", "sha256"},
            "raw activation fingerprint missing fields")
    shape = value["shape"]
    require(type(shape) is list and bool(shape) and all(type(n) is int and n >= 0 for n in shape)
            and type(value["dtype"]) is str and type(value["bytes"]) is int and value["bytes"] >= 0,
            "invalid activation geometry fingerprint")
    validate_digest(value["sha256"], "activation bytes")


def projection_identity(control):
    """Independently bind all nonempty split/gather checks to native layouts."""
    swa=control.get("swa_events")
    events=control.get("projection_identity_events")
    require(type(swa) is list and type(events) is list, "split projection raw evidence required")
    expected={(s,s+8192,l) for s in (0,8192) for l in FFN_LAYERS}
    by_key={}
    for event in swa:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in by_key,"qualification SWA coverage mismatch")
        positions=_positions(event.get("current_positions"),key[0],key[1],full_required=key[2]<20)
        require(event.get("full_rows")==len(positions) and event.get("online_rows")==len(positions)
                and event.get("online_local_rows")==list(range(len(positions)))
                and event.get("reused_rows")==0,"qualification must recompute every native SWA row")
        by_key[key]=event
    require(set(by_key)==expected,"qualification missing layer/chunk SWA evidence")
    nonempty={k for k,v in by_key.items() if v["full_rows"]>0}
    seen=set()
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        names=["split_q_a","split_swa","split_main_q_b","packed_gather_swa127"]
        if key[2] in (8,14,20,24,28,32,36): names.append("split_index_q_b")
        require(key in nonempty and key not in seen and event.get("full_rows")==by_key[key]["full_rows"]
                and event.get("checks")==len(names) and type(event.get("max_absolute_error")) in (int,float)
                and event["max_absolute_error"]==0
                and event.get("checks_named")==names,
                "split Q/KV/gather exact-zero checks missing or failed")
        seen.add(key)
    count=sum(5 if key[2] in (8,14,20,24,28,32,36) else 4 for key in nonempty)
    require(count>0 and seen==nonempty and control.get("projection_identity_checks")==count,
            "projection qualification count/coverage mismatch")
    return count


def cache_restore_identity(control):
    """Every nonempty native layout must prove exact canonical/RoPE/cast assembly."""
    source = control.get("swa_events")
    events = control.get("cache_restore_identity_events")
    require(type(source) is list and type(events) is list, "cache restoration numerical evidence missing")
    expected = {(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in source if e["full_rows"]>0}
    seen=set()
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen and event.get("full_rows")==expected[key]["full_rows"]
                and event.get("checks")==2 and type(event.get("max_absolute_error")) in (int,float)
                and event["max_absolute_error"]==0
                and event.get("checks_named")==["canonical_norm_rope_cast","assembled_swa_rows"],
                "cache normalization/RoPE/cast/assembly qualification is not exact")
        seen.add(key)
    count=2*len(expected)
    require(count>0 and seen==set(expected) and control.get("cache_restore_identity_checks")==count,
            "cache restoration numerical coverage/count mismatch")
    return count


def audit_joint_attention(control):
    """Audit actual selected query execution, not merely cached-row counts."""
    source=control.get("swa_events");events=control.get("attention_events")
    require(type(source) is list and type(events) is list,"SWA/attention raw accounting absent")
    expected={(e["chunk_start"],e["chunk_end"],e["layer"]):e for e in source}
    require(len(expected)==len(source),"duplicate SWA event")
    seen=set();reused=0
    dense=control.get("mode")=="reference_sparse"
    for event in events:
        key=(event.get("chunk_start"),event.get("chunk_end"),event.get("layer"))
        require(key in expected and key not in seen,"attention coverage differs")
        seen.add(key);swa=expected[key];n=swa["full_rows"];k=swa["online_rows"];layer=key[2]
        local=layer in (8,14,20,24,28,32,36)
        expected_actual=n if dense else k
        require(event.get("full_rows")==n and event.get("selected_rows")==k
                and event.get("main_q_a_rows")== (n if local or dense else k)
                and event.get("index_q_rows")== (n if local else 0)
                and event.get("main_q_b_rows")==expected_actual
                and event.get("joint_attention_rows")==expected_actual
                and event.get("output_projection_rows")==expected_actual
                and event.get("inactive_attention_zero_rows")==n-k
                and event.get("reference_dense") is dense
                and event.get("global_compute_called") is True
                and event.get("native_joint_attention") is True
                and event.get("is_global_source") is (layer in (8,14,20)),
                "actual Q/attention/O execution differs from B selection")
        require(type(event.get("global_kv_rows")) is int and event["global_kv_rows"]>=0,
                "complete global KV geometry missing")
        validate_digest(event.get("plan_sha256"),"attention plan")
        reused+=swa["reused_rows"]
    require(seen==set(expected),"attention layers missing")
    return reused


def audit_native_no_cache(control):
    require(all(control.get(k)==[] for k in
                ("selection","index_groups","swa_events","ffn_layouts","attention_events",
                 "projection_identity_events","cache_restore_identity_events"))
            and control.get("projection_identity_checks")==0
            and control.get("cache_restore_identity_checks")==0
            and control.get("adapter",{}).get("events")==[],
            "native/offline phase cannot claim sparse or cached intervention")


def cuda_mask_qualification(directory, preflight, evidence):
    from gpu_pilot.cuda_mask_qualification import validate_report
    value = read(directory/"cuda_mask_qualification.json", evidence)
    require(value.get("schema") == "ffn-only-cuda-mask-all-ranks-v1" and value.get("passed") is True
            and value.get("execution_configuration_sha256") == preflight["identity"]["execution_configuration_sha256"]
            and value.get("quality_eligible") is False and value.get("timing_valid") is False,
            "CUDA mask qualification not bound to this execution")
    reports = value.get("reports")
    require(type(reports) is list and len(reports) == 8
            and [r.get("rank") for r in reports] == list(range(8)), "all eight CUDA rank checks required")
    for item in reports:
        validate_report(item["report"])
    return value


def qualification(directory, preflight, evidence):
    mask = cuda_mask_qualification(directory, preflight, evidence)
    from gpu_pilot.index_hit20_empty_ep_qualification import validate_all_ranks_report
    empty_ep = read(directory/"empty_ep_qualification.json", evidence)
    validate_all_ranks_report(empty_ep,
        execution_sha256=preflight["identity"]["execution_configuration_sha256"],
        model_sha256=preflight["identity"]["checkpoint_index_sha256"])
    raw = [read(directory/name, evidence) for name in
           ("qualification.native.json", "qualification.all_selected.json")]
    closure = read(directory/"qualification.json", evidence)
    require(closure.get("cuda_mask_qualification_sha256") == mask["sha256"],
            "strict numerical qualification must bind completed CUDA mask checks")
    require(closure.get("empty_ep_qualification_sha256") == empty_ep["sha256"],
            "strict numerical qualification must bind current native empty-EP proof")
    require(closure.get("schema") == "tail-link-qualification-v1" and closure.get("passed") is True
            and closure.get("reference_sha256") == raw[0]["sha256"]
            and closure.get("candidate_sha256") == raw[1]["sha256"]
            and closure.get("qualification_input_tokens") == 16384
            and closure.get("identity_check_count") == 80
            and closure.get("full_256k_numerical_parity_claim") is False
            and closure.get("old_qualification_reused") is False, "qualification closure mismatch")
    fingerprints = []
    for phase, mode, name in zip(raw, ("qualified_native", "all_selected"),
                                 ("qualification.native","qualification.all_selected")):
        empty_ep_binding(phase, closure)
        require(phase.get("schema") == PHASE_SCHEMA and phase.get("mode") == mode
                and phase.get("index") == 0 and phase.get("arm") == "qualification"
                and phase.get("name") == name and phase.get("repeat") is None
                and phase.get("timing_valid") is False and phase.get("quality_eligible") is False,
                "foreign qualification phase")
        binding(phase, preflight)
        record, control = phase["record"], phase["controller"]
        if mode == "qualified_native":
            audit_native_no_cache(control)
        expected_input=preflight.get("qualification_input_ids_sha256")
        validate_digest(expected_input,"qualification manifest prefix")
        require(record.get("success") is True and record.get("rank") == 0
                and phase.get("input_ids_sha256")==record.get("input_ids_sha256")==control.get("input_ids_sha256")==expected_input
                and control.get("full_input_tokens")==16384
                and record.get("phase")==control.get("phase")==name and control.get("mode")==mode
                and record.get("natural_EOS") is False
                and record.get("input_tokens") == 16384 and len(record.get("output_ids", [])) == 1
                and record.get("output_ids_sha256") == digest(record["output_ids"]),
                "qualification output/length mismatch")
        rows = control.get("identity_fingerprints")
        expected = {(start, start+8192, layer) for start in (0, 8192) for layer in range(40)}
        require(type(rows) is list and len(rows) == 80 and control.get("identity_check_count") == 80,
                "all eighty raw layer/chunk fingerprints required")
        seen = set()
        for row in rows:
            key = tuple(row.get("key", []))
            require(key in expected and key not in seen, "foreign or duplicate qualification layer/chunk")
            seen.add(key)
            value = row["value"]
            require(set(value) == {"output", "lazy"}
                    and set(value["lazy"]) == {"residual", "post_layer_mix", "comb_res_mix", "pre_layer_mix"},
                    "every native lazy field must be fingerprinted")
            require(value["output"] is not None, "decoder output missing")
            _fingerprint(value["output"])
            for item in value["lazy"].values():
                _fingerprint(item)
        require(digest(rows) == control.get("identity_fingerprints_sha256"), "fingerprint aggregate mismatch")
        fingerprints.append(rows)
    require(raw[0]["input_ids_sha256"] == raw[1]["input_ids_sha256"]
            and raw[0]["record"]["output_ids"] == raw[1]["record"]["output_ids"]
            and fingerprints[0] == fingerprints[1]
            and raw[1]["controller"].get("identity_exact") is True, "all-selected differs from native")
    checks=projection_identity(raw[1]["controller"])
    require(closure.get("projection_identity_checks")==checks,"split qualification closure mismatch")
    require(closure.get("cache_restore_identity_checks")==cache_restore_identity(raw[1]["controller"]),
            "cache restoration qualification closure mismatch")
    require(audit_joint_attention(raw[1]["controller"])==0,"all-selected qualification cannot reuse cached rows")
    return closure


def empty_ep_binding(phase, qualification):
    proof = qualification.get("empty_ep_qualification_sha256")
    validate_digest(proof, "native empty-EP qualification")
    require(phase.get("empty_ep_qualification_sha256") == proof
            and phase.get("controller", {}).get("empty_ep_qualification_sha256") == proof,
            "phase/controller missing or foreign native empty-EP capability")


def audit_native_swa_configuration(preflight):
    """Bind representation to the verified frozen native configuration.

    YOCO row compaction does not imply shared BF16 SWA projection input. This
    protocol fixes the native flag to False, and binds the preflight parser /
    checkpoint proof to the independently hashed native source inventory.
    """
    identity = preflight.get("identity", {})
    proof = identity.get("native_yoco_swa_configuration", {})
    require(preflight.get("native_yoco_swa_share_hidden") is False
            and preflight.get("protocol", {}).get("native_yoco_swa_share_hidden") is False
            and identity.get("native_yoco_swa_share_hidden") is False
            and proof.get("schema") == "tail-link-native-yoco-swa-configuration-v1"
            and proof.get("native_yoco_swa_share_hidden") is False
            and proof.get("hf_default") is False
            and proof.get("parser_override_absent") is True
            and proof.get("verified") is True,
            "native SWA configuration must verify native_yoco_swa_share_hidden=False")
    parser = "deepinfer/hugging_face/config.py"
    require(proof.get("hf_parser_file") == parser,
            "native SWA configuration proof names a different HF parser")
    for key, expected in (("hf_parser_sha256", identity.get("runtime_source_files_sha256", {}).get(parser)),
                          ("checkpoint_config_sha256", identity.get("checkpoint_config_sha256"))):
        validate_digest(proof.get(key), "native SWA configuration "+key)
        require(proof[key] == expected, "native SWA configuration hash drift: "+key)
    overrides = proof.get("config_overrides")
    require(type(overrides) is list and all(type(item) is dict
            and type(item.get("path")) is str and item["path"].startswith("/")
            and item["path"].endswith("/yoco_swa_share_hidden")
            and item.get("value") is False for item in overrides),
            "checkpoint SWA configuration overrides must be explicit False")
    require(len({item["path"] for item in overrides}) == len(overrides),
            "duplicate checkpoint SWA configuration override")
    return False


def binding(phase, preflight):
    audit_native_swa_configuration(preflight)
    require(phase.get("manifest_sha256") == preflight["manifest_sha256"]
            and phase.get("tail_link_protocol_sha256") == preflight["protocol"]["sha256"]
            and phase.get("execution_configuration_sha256") ==
                preflight["identity"]["execution_configuration_sha256"], "phase execution identity drift")
    require(phase.get("native_yoco_swa_share_hidden") is False
            and phase.get("controller", {}).get("native_yoco_swa_share_hidden") is False,
            "phase/controller native_yoco_swa_share_hidden must match verified False configuration")


def _positions(value, start, end, *, full_required=False):
    require(type(value) is list and all(type(p) is int and start <= p < end for p in value)
            and value == sorted(set(value)), "native row positions must be unique, sorted, absolute and in chunk")
    if full_required:
        require(value == list(range(start,end)), "encoder native row layout unexpectedly compacted")
    return value


def audit_staging(value, spans):
    require(type(value) is dict and value.get("schema") == "tail-link-h2d-staging-v1"
            and type(value.get("phase_generation")) is int and value["phase_generation"] > 0
            and value.get("double_buffers") == 2 and value.get("cpu_pinned") is True
            and value.get("cross_phase_gpu_reuse") is False and value.get("copies_after_reset") is True
            and value.get("same_phase_restored_reuse") is True
            and value.get("copy_ms_is_not_wall_or_ttft") is True
            and value.get("document_keys") == [key for _, _, key in spans],
            "CPU tail cache/staging generation or document binding invalid")
    copies = value.get("copies")
    require(type(copies) is list and len(copies) == 37, "one fresh tail H2D per post3 layer required")
    seen = set()
    for copy in copies:
        layer = copy.get("layer")
        require(type(layer) is int and layer in FFN_LAYERS and layer not in seen
                and copy.get("slot") in (0, 1) and type(copy.get("slot")) is int
                and copy.get("generation") == value["phase_generation"]
                and copy.get("bytes") == 8*128*512*2, "tail DMA identity/size mismatch")
        number(copy.get("cuda_copy_ms"), "CUDA H2D event duration")
        seen.add(layer)
    total = sum(item["bytes"] for item in copies)
    require(seen == set(FFN_LAYERS) and value.get("copy_count") == 37
            and value.get("copy_bytes") == total, "tail DMA coverage or total mismatch")
    return total


def _event_map(events, expected, label):
    require(type(events) is list, label+" events absent")
    result = {}
    for event in events:
        key = (event.get("chunk_start", event.get("start")),
               event.get("chunk_end", event.get("end")), event.get("layer"))
        require(key in expected and key not in result, label+" layer/chunk duplicate or foreign")
        result[key] = event
    require(set(result) == expected, label+" layer/chunk coverage incomplete")
    return result


def audit_memory_event(event, positions, selected, spans, available_non_document, dense):
    """Reconstruct causal keys from absolute positions, never trust producer counts."""
    start = event.get("chunk_start", event.get("start"))
    end = event.get("chunk_end", event.get("end"))
    query_positions = positions if dense else [positions[i] for i in selected]
    available = sorted(available_non_document)
    require(event.get("query_positions") == query_positions
            and event.get("available_nondoc_positions") == available,
            "memory queries or actually generated non-document keys differ")
    # An offline document key is allowed only at its real input position.
    last_query = positions[-1] if positions else -1
    tail = {p: (i, p-lo, p-(hi-128)) for i, (lo, hi, _) in enumerate(spans)
            for p in range(hi-128, min(hi, last_query+1))}
    expected = sorted(set(tail) | {p for p in available if p <= last_query})
    require(event.get("memory_positions") == expected, "tail memory includes missing/future/non-tail key")
    require(event.get("query_memory_lengths") == [bisect_right(expected, q) for q in query_positions]
            and event.get("actual_joint_query_rows") == len(query_positions),
            "per-query causal memory support or executed queries differ")
    require(event.get("memory_doc_indices") == [tail[p][0] if p in tail else -1 for p in expected]
            and event.get("memory_doc_token_offsets") == [tail[p][1] if p in tail else -1 for p in expected]
            and event.get("memory_tail_bank_rows") == [tail[p][2] if p in tail else -1 for p in expected]
            and event.get("memory_kinds") == ["document_tail" if p in tail else "online_nondocument" for p in expected],
            "tail source-document/local-offset mapping differs")
    reference_lengths = [bisect_right(expected, q) for q in positions]
    owner = {p: i for i, p in enumerate(positions)}
    require(event.get("memory_count") == len(expected)
            and event.get("memory_current_local_rows") == [owner.get(p, -1) for p in expected]
            and event.get("selected_memory_lengths") == [reference_lengths[i] for i in selected]
            and event.get("reference_memory_lengths") == reference_lengths
            and event.get("global_index_offset") == len(expected)
            and event.get("compact_slots_are_absolute_positions") is False
            and event.get("nondocument_availability") == "explicit_generated_only",
            "tail compact-index/global-offset causal mapping differs")
    normalized = [dict(occurrence=i, key=key, start=lo, end=hi,
                       tail_start=hi-128, tail_end=hi, tail_rows=128)
                  for i, (lo, hi, key) in enumerate(spans)]
    non_doc = [i for i, p in enumerate(positions) if not any(lo <= p < hi for lo, hi, _ in spans)]
    tail_ids = [i for i in selected if i not in set(non_doc)]
    plan = dict(schema="kce-tail-link-plan-v1", full_input_tokens=event.get("full_input_tokens"),
        chunk_start=start, chunk_end=end, tail_tokens=128, documents=normalized,
        current_positions=positions, selected_local_rows=selected,
        selected_positions=[positions[i] for i in selected], document_tail_local_rows=tail_ids,
        mandatory_nondoc_local_rows=non_doc, inactive_local_rows=[i for i in range(len(positions)) if i not in set(selected)],
        selected_rows=len(selected), full_rows=len(positions), memory_positions=expected,
        memory_count=len(expected), memory_kinds=["document_tail" if p in tail else "online_nondocument" for p in expected],
        memory_doc_indices=[tail[p][0] if p in tail else -1 for p in expected],
        memory_doc_token_offsets=[tail[p][1] if p in tail else -1 for p in expected],
        memory_tail_bank_rows=[tail[p][2] if p in tail else -1 for p in expected],
        memory_current_local_rows=[owner.get(p, -1) for p in expected],
        selected_memory_lengths=[reference_lengths[i] for i in selected], reference_memory_lengths=reference_lengths,
        available_nondoc_positions=available, nondocument_availability="explicit_generated_only",
        compact_slots_are_absolute_positions=False, global_index_offset=len(expected),
        native_yoco_surviving_positions_preserved=True)
    require(all(event.get(k) == value for k, value in plan.items())
            and event.get("plan_sha256") == digest(plan), "tail plan seal differs from independently derived map")
    return len(tail), len(expected)-len(tail)


def audit_tail_restore_qualification(control, expected, spans):
    """Raw DMA/absolute-RoPE and assembled memory checks, not policy parity alone."""
    events = control.get("tail_restore_identity_events")
    require(type(events) is list and len(events) == 37, "all post3 tail DMA/restore checks required")
    seen = set()
    for event in events:
        layer = event.get("layer")
        require(layer in FFN_LAYERS and layer not in seen and event.get("checks") == 2
                and event.get("checks_named") == ["dma_bytes", "absolute_rope_nativecast"]
                and type(event.get("max_absolute_error")) in (float, int)
                and event["max_absolute_error"] == 0 and event.get("rows") == 1024
                and event.get("absolute_positions_sha256") == digest([p for _,hi,_ in spans for p in range(hi-128,hi)]),
                "tail DMA/RoPE exact-zero check missing")
        seen.add(layer)
    assembly = _event_map(control.get("memory_assembly_identity_events"), expected, "memory assembly oracle")
    memory = _event_map(control.get("tail_memory_events"), expected, "tail memory")
    for key, event in assembly.items():
        require(event.get("schema") == "tail-link-memory-assembly-identity-v1"
                and event.get("checks") == 1 and type(event.get("max_absolute_error")) in (float, int)
                and event["max_absolute_error"] == 0 and event.get("rows") == memory[key].get("memory_count"),
                "tail assembled memory oracle failed")



def audit_dynamic_swa_event(event, positions, tail_ids, spans, accumulated, *, generation, qualification,
                            native_yoco_swa_share_hidden=False):
    """Independent per-layer online refresh and explicit YOCO fallback audit.

    ``accumulated`` is owned by the reader and starts empty for each phase and
    layer. It is populated only by raw current native-surviving tail positions.
    Producer accumulated/fallback lists cannot add hidden states or revive rows.
    """
    layer = event.get("layer")
    start, end = event.get("chunk_start"), event.get("chunk_end")
    require(type(layer) is int and layer in FFN_LAYERS
            and type(start) is int and type(end) is int,
            "dynamic SWA event layer/chunk invalid")
    require(type(native_yoco_swa_share_hidden) is bool,
            "dynamic SWA native configuration flag must be an explicit bool")
    _positions(positions, start, end)
    enabled = layer >= 4
    expected_ids = tail_ids if enabled else []
    current_tail = [positions[i] for i in tail_ids]
    refresh_positions = [positions[i] for i in expected_ids]
    tail_slots = {p: ordinal*128+p-(hi-128)
                  for ordinal, (_, hi, _) in enumerate(spans) for p in range(hi-128, hi)}
    require(all(p in tail_slots for p in current_tail)
            and not (set(refresh_positions) & accumulated), "dynamic SWA duplicate/non-tail refresh")
    next_accumulated = accumulated | set(refresh_positions)
    cutoff = positions[-1] if positions else -1
    visible_tails = {p for p in tail_slots if p <= cutoff}
    visible_refreshed = visible_tails & next_accumulated
    expected = dict(schema="tail-link-dynamic-swa-v1", phase_generation=generation,
        refresh_enabled=enabled,
        source="current_layer_attention_input" if enabled else "offline_layer3_seed",
        input_representation=("bf16_current" if native_yoco_swa_share_hidden and layer >= 20
                              else "native_packed_current"),
        current_tail_local_rows=tail_ids, current_tail_positions=current_tail,
        refreshed_local_rows=expected_ids, refreshed_positions=refresh_positions,
        refreshed_bank_rows=[tail_slots[p] for p in refresh_positions],
        available_refreshed_positions=sorted(next_accumulated),
        visible_refreshed_positions=sorted(visible_refreshed),
        visible_offline_fallback_positions=sorted(visible_tails-next_accumulated))
    context = f"dynamic SWA layer={layer} chunk=[{start},{end}) "
    require(type(event.get("phase_generation")) is int and generation > 0,
            context+"phase_generation must be a positive integer")
    for field, value in expected.items():
        matches = event.get(field) is value if type(value) is bool else event.get(field) == value
        require(matches, context+field+" differs from independently derived expectation")
    checked = qualification and bool(expected_ids)
    require(event.get("projection_identity_checked") is checked
            and event.get("prepare_identity_checked") is checked
            and event.get("checks") == (2 if checked else 0)
            and ((type(event.get("max_absolute_error")) in (float, int)
                  and event["max_absolute_error"] == 0) if checked else event.get("max_absolute_error") is None),
            context+"exact projection/normalization/RoPE/cast proof incomplete")
    accumulated.update(refresh_positions)
    return dict(refreshed_rows=len(refresh_positions),
                visible_refreshed_rows=len(visible_refreshed),
                visible_offline_fallback_rows=len(visible_tails-next_accumulated))


def audit_controller(control, row, phase, spans):
    require(control.get("native_yoco_swa_share_hidden") is False
            and phase.get("native_yoco_swa_share_hidden") is False,
            "phase/controller native_yoco_swa_share_hidden must match verified False configuration")
    require(control.get("schema") == "tail-link-controller-v1"
            and control.get("mode") == phase["mode"] and control.get("phase") == phase["name"]
            and control.get("input_ids_sha256") == row["input_ids_sha256"]
            and control.get("full_input_tokens") == row["input_tokens"]
            and control.get("ffn_layers") == list(FFN_LAYERS)
            and all(control.get(k) is True for k in ("first_three_full", "actual_full_row_native_mhc",
                "yoco_layout_preserved", "formal_admission_unchanged", "decode_native",
                "global_kv_native_online", "not_model_flops_fraction", "prefill_only_link"))
            and control.get("selected_tokens_forced_through_yoco") is False,
            "tail controller scope/safeguards differ")
    adapter = control.get("adapter", {})
    counts = adapter.get("counts", {})
    require(adapter.get("schema") == "kv-cache-engram-index-hit20-adapter-v1"
            and all(adapter.get(k) is True for k in ("decoder_forward_unchanged", "attention_forward_unchanged",
                "native_mhc_unchanged", "native_yoco_layout_unchanged", "all_selected_delegates_native"))
            and all(adapter.get(k) is False for k in ("empty_selection_skips_ep_collective",
                "cached_hidden_or_ffn_output", "accuracy_guarantee", "speedup_claim")),
            "underlying native/empty-EP FFN adapter safeguards changed")
    dynamic = phase["arm"] == "tail128_dynamic"
    require(control.get("dynamic_document_swa") is dynamic
            and control.get("dynamic_refresh_start_layer") == (4 if dynamic else None),
            "document SWA refresh policy differs from arm")
    if not dynamic:
        require(control.get("dynamic_swa_events") == [], "static/native arm contains dynamic refresh evidence")
    if phase["arm"] == "recomputed":
        audit_native_no_cache(control)
        require(control.get("staging") is None and counts.get("full_rows") == 0
                and counts.get("selected_rows") == 0, "baseline cannot claim cache intervention")
        return dict(ffn_full_rows=0, ffn_selected_rows=0, ffn_bypassed_rows=0,
                    swa_full_rows=0, swa_online_rows=0, swa_reused_rows=0,
                    attention_full_rows=0, attention_selected_rows=0, h2d_bytes=0)
    require(phase["arm"] in ARMS[1:] and phase["mode"] in ("sensitive", "reference_sparse")
            and phase.get("fraction") is None, "unknown tail policy")
    h2d_bytes = audit_staging(control.get("staging"), spans)
    chunks = [(s, min(s+8192, row["input_tokens"])) for s in range(0, row["input_tokens"], 8192)]
    expected = {(s, e, layer) for s, e in chunks for layer in FFN_LAYERS}
    masks = _event_map(control.get("selection"), expected, "selection")
    layouts = _event_map(control.get("ffn_layouts"), expected, "FFN layout")
    ffn = _event_map(adapter.get("events"), expected, "FFN execution")
    swa = _event_map(control.get("swa_events"), expected, "SWA")
    attn = _event_map(control.get("attention_events"), expected, "attention")
    memories = _event_map(control.get("tail_memory_events"), expected, "tail memory")
    refreshes = _event_map(control.get("dynamic_swa_events"), expected, "dynamic SWA") if dynamic else {}
    refreshed = {layer: set() for layer in FFN_LAYERS}
    dense = phase["mode"] == "reference_sparse"
    if dense or phase.get("verify_sparse") is True:
        audit_tail_restore_qualification(control, expected, spans)
    full = selected = actual_ffn = reused = online = memory_tail = memory_online = actual_attention = 0
    memory_tail_pairs = memory_online_pairs = 0
    dynamic_rows = dynamic_materialized = dynamic_pairs = 0
    compact = {}; available = {layer: set() for layer in FFN_LAYERS}
    encoder = encoder_selected = decoder = decoder_selected = partial = 0
    for key in sorted(expected):
        start, end, layer = key
        m, layout, fe, se, ae = masks[key], layouts[key], ffn[key], swa[key], attn[key]
        positions = _positions(m.get("current_positions"), start, end, full_required=layer < 20)
        if layer >= 20:
            require(compact.setdefault((start, end), positions) == positions, "YOCO survivor layout changed")
        ids, non_doc, tail_ids = mask_reference(positions, start, end, spans)
        n, k = len(positions), len(ids)
        validate_digest(m.get("plan_sha256"), "tail plan")
        require(m.get("selected_local_rows") == ids and m.get("selected_rows") == k
                and m.get("full_rows") == n and m.get("selected_positions") == [positions[i] for i in ids],
                "selected rows differ from independent tail128/non-document policy")
        require(all(layout.get(field) == m.get(field) for field in
                    ("current_positions", "selected_local_rows", "full_rows", "selected_rows", "plan_sha256")),
                "FFN layout differs from attention selection")
        actual = n if dense else k
        require(fe.get("full_rows") == n and fe.get("selected_rows") == k
                and fe.get("actual_mlp_rows") == actual and fe.get("reference_dense") is dense
                and fe.get("plan_sha256") == m["plan_sha256"]
                and fe.get("request_id") == phase["record"]["request_id"] and fe.get("phase") == phase["name"]
                and fe.get("inactive_ffn_output") == "zero"
                and fe.get("empty_native_collective") is (k == 0 and n > 0),
                "actual FFN execution differs from tail selection")
        local_index = layer in (8, 14, 20, 24, 28, 32, 36)
        require(ae.get("full_rows") == n and ae.get("selected_rows") == k
                and ae.get("main_q_a_rows") == (n if local_index or dense else k)
                and ae.get("index_q_rows") == (n if local_index else 0)
                and ae.get("main_q_b_rows") == actual and ae.get("joint_attention_rows") == actual
                and ae.get("output_projection_rows") == actual
                and ae.get("global_compute_called") is True and ae.get("native_joint_attention") is True
                and ae.get("reference_dense") is dense
                and ae.get("is_global_source") is (layer in (8, 14, 20))
                and ae.get("plan_sha256") == m["plan_sha256"],
                "global/Q/joint attention actual row path differs")
        require(type(ae.get("global_kv_rows")) is int and ae["global_kv_rows"] >= 0,
                "native complete global KV geometry absent")
        refresh_ids = tail_ids if dynamic and layer >= 4 else []
        if dynamic:
            audit_dynamic_swa_event(refreshes[key], positions, tail_ids, spans,
                refreshed[layer], generation=control["staging"]["phase_generation"],
                qualification=dense or phase.get("verify_sparse") is True,
                native_yoco_swa_share_hidden=control["native_yoco_swa_share_hidden"])
        online_ids = sorted(non_doc + refresh_ids)
        require(se.get("current_positions") == positions and se.get("full_rows") == n
                and se.get("document_online_rows") == len(refresh_ids)
                and se.get("document_tail_rows") == len(tail_ids)
                and se.get("non_document_online_rows") == len(non_doc)
                and se.get("reused_rows") == len(tail_ids)-len(refresh_ids)
                and se.get("online_local_rows") == online_ids and se.get("online_rows") == len(online_ids)
                and se.get("compact") is (layer >= 20), "SWA tail/online actual row accounting differs")
        available[layer].update(positions[i] for i in non_doc)
        mt, mo = audit_memory_event(memories[key], positions, ids, spans, available[layer], dense)
        require(memories[key].get("plan_sha256") == m["plan_sha256"], "memory detached from selection plan")
        require(memories[key].get("full_input_tokens") == row["input_tokens"]
                and memories[key].get("gpu_causal_oracle_checked") is (dense or phase.get("verify_sparse") is True),
                "memory full input or raw GPU causal oracle eligibility differs")
        visible_refreshed = set(memories[key]["memory_positions"]) & refreshed[layer]
        memory_tail += mt-len(visible_refreshed); memory_online += mo
        dynamic_rows += len(refresh_ids); dynamic_materialized += len(visible_refreshed)
        prefix, dynamic_prefix = [0], [0]
        for position, kind in zip(memories[key]["memory_positions"], memories[key]["memory_kinds"]):
            prefix.append(prefix[-1]+(kind == "document_tail" and position not in refreshed[layer]))
            dynamic_prefix.append(dynamic_prefix[-1]+(position in refreshed[layer]))
        # Query-key pairs really consumed by this arm, not allocated memory rows.
        lengths = memories[key]["query_memory_lengths"]
        memory_tail_pairs += sum(prefix[length] for length in lengths)
        memory_online_pairs += sum(length-prefix[length]-dynamic_prefix[length] for length in lengths)
        dynamic_pairs += sum(dynamic_prefix[length] for length in lengths)
        full += n; selected += k; actual_ffn += actual; actual_attention += actual
        reused += len(tail_ids)-len(refresh_ids); online += len(online_ids); partial += 0 < k < n
        if layer < 20:
            encoder += n; encoder_selected += k
        else:
            decoder += n; decoder_selected += k
    require(counts.get("full_rows") == full and counts.get("selected_rows") == selected
            and counts.get("actual_mlp_rows") == actual_ffn
            and counts.get("inactive_ffn_zero_rows") == full-selected, "FFN aggregate differs")
    rings = control.get("decode_ring_positions")
    require(type(rings) is list and len(rings) == len(FFN_LAYERS)
            and {x.get("layer") for x in rings} == set(FFN_LAYERS)
            and all(x.get("positions") == list(range(row["input_tokens"]-128, row["input_tokens"]))
                    and x.get("physical_readback_exact") is True and x.get("checks") == 128
                    and type(x.get("physical_slots")) is list and len(x["physical_slots"]) == 128
                    and len(set(x["physical_slots"])) == 128
                    and all(type(slot) is int and slot >= 0 for slot in x["physical_slots"]) for x in rings),
            "final native decode ring is missing last128 KV")
    return dict(ffn_full_rows=full, ffn_selected_rows=selected, ffn_bypassed_rows=full-selected,
                ffn_actual_computed_rows=actual_ffn, attention_full_rows=full,
                attention_selected_rows=selected, attention_bypassed_rows=full-selected,
                attention_actual_computed_rows=actual_attention, swa_full_rows=reused+online,
                swa_online_rows=online, swa_reused_rows=reused,
                swa_reuse_fraction=reused/(reused+online) if reused+online else 0.,
                materialized_memory_cached_tail_rows=memory_tail,
                materialized_memory_online_non_document_rows=memory_online,
                consumed_memory_cached_tail_pairs=memory_tail_pairs,
                consumed_memory_online_non_document_pairs=memory_online_pairs,
                document_swa_refreshed_rows=dynamic_rows, dynamic_refreshed_rows=dynamic_rows,
                dynamic_fallback_memory_rows=memory_tail if dynamic else 0,
                materialized_memory_online_document_rows=dynamic_materialized,
                consumed_memory_online_document_pairs=dynamic_pairs,
                dynamic_refresh_start_layer=4 if dynamic else None,
                dynamic_partial_yoco_fallback=dynamic,
                dynamic_tail_slot_capacity=sum(layer >= 4 for layer in FFN_LAYERS)*len(spans)*128 if dynamic else 0,
                dynamic_refreshed_tail_slot_fraction=dynamic_rows/(sum(layer >= 4 for layer in FFN_LAYERS)*len(spans)*128) if dynamic else None,
                dynamic_unrefreshed_yoco_tail_slots=sum(layer >= 4 for layer in FFN_LAYERS)*len(spans)*128-dynamic_rows if dynamic else 0,
                dynamic_seed_tail_slots=len(spans)*128 if dynamic else 0,
                encoder_rows=encoder, encoder_selected_rows=encoder_selected,
                decoder_rows=decoder, decoder_selected_rows=decoder_selected,
                actual_partial_chunks=partial, h2d_bytes=h2d_bytes,
                first_three_layers_excluded=True, not_model_flops_fraction=True)

def audit_phase(phase, row, preflight, *, arm, timing, quality, fixed_tokens=32):
    sealed(phase)
    binding(phase, preflight)
    require(phase.get("schema") == PHASE_SCHEMA and phase.get("index") == row["index"]
            and phase.get("dataset") == row["dataset"] and phase.get("arm") == arm
            and phase.get("dynamic_document_swa") is (arm == "tail128_dynamic")
            and phase.get("mode") == ("native" if arm == "recomputed" else "sensitive")
            and phase.get("fraction") == FRACTIONS[arm] and phase.get("timing_valid") is timing
            and phase.get("quality_eligible") is quality and phase.get("input_ids_sha256") == row["input_ids_sha256"]
            and phase.get("online_cpu_tail_prefetch_included") is True
            and phase.get("online_clock_started_before_reset") is True
            and phase.get("phase_local_gpu_staging") is True
            and phase.get("offline_build_included") is False
            and phase.get("native_global_kv_and_index_preserved") is True, "phase scope mismatch")
    require(phase.get("partial_smoke") is (not quality and not timing and fixed_tokens == 1),
            "partial smoke must remain separate from quality and formal timing")
    record = phase["record"]
    require(record.get("success") is True and record.get("rank") == 0
            and record.get("input_ids_sha256") == row["input_ids_sha256"]
            and record.get("input_tokens") == row["input_tokens"]
            and record.get("phase") == phase["name"] and record.get("sample_id") == row["sample_id"]
            and record.get("dataset") == row["dataset"]
            and record.get("request_id") == f"tail-link:{row['index']}:{phase['name']}:{row['sample_id']}",
            "complete request scope mismatch")
    ids = record.get("output_ids")
    require(type(ids) is list and bool(ids) and all(type(x) is int and x >= 0 for x in ids)
            and digest(ids) == record.get("output_ids_sha256"), "output token evidence mismatch")
    if quality:
        require(record.get("natural_EOS") is True and len(ids) <= 64
                and record.get("finish_reason") in ("length", "stop")
                and record.get("truncated") is (record.get("finish_reason") == "length"), "quality EOS/cap mismatch")
        actual = score(record.get("text"), row["references"])
        require(all(type(record.get(key)) in (float, int) and math.isfinite(record[key])
                    and abs(record[key]-value) <= 1e-12 for key, value in actual.items()),
                "quality score not independently reproducible against fixed gold")
    else:
        require(type(fixed_tokens) is int and fixed_tokens in (1, 32)
                and (not timing or fixed_tokens == 32)
                and len(ids) == fixed_tokens and record.get("natural_EOS") is False
                and record.get("finish_reason") == "length"
                and record.get("truncated") is False, "wrong fixed output length or EOS policy")
    ttft = number(record.get("online_ttft_s"), "online TTFT", True)
    elapsed = number(record.get("online_elapsed_s"), "online elapsed", True)
    prepared = number(record.get("online_prepare_to_submit_s"), "online preparation")
    require(prepared <= ttft <= elapsed <= number(phase.get("online_global_window_s"), "common wall", True),
            "invalid online timing ordering")
    return audit_controller(phase["controller"], row, phase, documents(row))



def partial_smoke(directory, preflight, qual, evidence, arm="tail128_link"):
    require(arm in ARMS[1:], "unknown smoke policy")
    prefix = "dynamic_" if arm == "tail128_dynamic" else ""
    row=next(item for item in preflight["audit_cases"] if item["index"]==2)
    closure=read(directory/(prefix+"partial_smoke.json"),evidence)
    require(closure.get("schema")=="tail-link-partial-smoke-v1" and closure.get("passed") is True
            and closure.get("index")==2 and closure.get("qualification_sha256")==qual["sha256"]
            and closure.get("input_ids_sha256")==row["input_ids_sha256"]
            and closure.get("input_tokens")==row["input_tokens"]
            and closure.get("quality_eligible") is False and closure.get("timing_valid") is False
            and closure.get("accuracy_guarantee") is False
            and closure.get("full_input_original_eight_documents") is True
            and closure.get("independent_swa_and_ffn_paths_exercised") is True
            and closure.get("output_tokens_per_arm")==1
            and set(closure.get("phase_sha256",{}))=={arm},"partial smoke closure incomplete")
    name=f"case2.{arm}_partial_smoke"
    phase=read(directory/(name+".json"),evidence)
    empty_ep_binding(phase, qual)
    require(phase.get("name")==name and phase.get("repeat") is None
            and phase["sha256"]==closure["phase_sha256"][arm],"partial smoke raw mismatch")
    rows=audit_phase(phase,row,preflight,arm=arm,timing=False,quality=False,fixed_tokens=1)
    sparse_reference=read(directory/f"case2.{arm}_sparse_reference.json",evidence)
    sparse_closure=read(directory/(prefix+"sparse_qualification.json"),evidence)
    binding(sparse_reference,preflight)
    empty_ep_binding(sparse_reference,qual)
    require(sparse_reference.get("name")==f"case2.{arm}_sparse_reference"
            and sparse_reference.get("repeat") is None and sparse_reference.get("partial_smoke") is False,
            "same-policy sparse reference phase scope mismatch")
    audit_controller(sparse_reference["controller"],row,sparse_reference,documents(row))
    expected_sparse=sparse_identity(sparse_reference,phase,row,qual["sha256"])
    require(sparse_closure==expected_sparse
            and closure.get("sparse_qualification_sha256")==sparse_closure["sha256"],
            "sparse same-policy numerical closure differs")
    raw_index_groups=0  # Position-based policy: no hit-count selection or raw-index replay claim.
    require(rows["ffn_bypassed_rows"]>0 and rows["swa_reused_rows"]>0
            and rows["actual_partial_chunks"]>0,"smoke did not exercise independent partial paths")
    return closure,dict(passed=True,arm=arm,index=2,input_tokens=row["input_tokens"],actual_rows=rows,
        closure_sha256=closure["sha256"],fixed_output_tokens=1,quality_eligible=False,timing_valid=False,
        first_three_fingerprints_sha256=digest([item for item in sparse_reference["controller"]["sparse_identity_fingerprints"]
                                               if item["key"][2] < 3]),
        numerical_identity_claim=False,same_policy_sparse_byteexact=True,
        sparse_qualification_sha256=sparse_closure["sha256"],raw_index_groups_audited=raw_index_groups,
        all_declared_layers_and_native_layouts_audited=True)


def audit_case(directory, row, preflight, qual, smoke, evidence, dynamic_smoke=None):
    index = row["index"]
    pair = read(directory/f"case{index}.paired.json", evidence)
    require(pair.get("schema") == "tail-link-case-paired-v1"
            and pair.get("qualification_sha256") == qual["sha256"]
            and pair.get("partial_smoke_sha256") == smoke["sha256"]
            and dynamic_smoke is not None and pair.get("dynamic_partial_smoke_sha256") == dynamic_smoke["sha256"]
            and all(pair.get(key) == row[key] for key in
                    ("index", "dataset", "sample_id", "references", "question", "input_tokens", "input_ids_sha256"))
            and pair.get("ffn_layers") == list(FFN_LAYERS)
            and pair.get("offline_capture_layers") == list(CAPTURE_LAYERS)
            and pair.get("all_negative_results_retained") is True
            and pair.get("exploratory_previously_exposed_inputs") is True, "three-arm closure identity mismatch")
    quality, timing, masks = {}, {arm: [] for arm in ARMS}, {}
    generations = set()
    def fresh_generation(phase):
        if phase["arm"] in ARMS[1:]:
            generation = phase["controller"]["staging"]["phase_generation"]
            require(generation not in generations, "warmup/quality/formal reused GPU staging generation")
            generations.add(generation)
    require(set(pair.get("quality", {})) == set(ARMS), "all three quality arms required")
    for arm in ARMS:
        phase = read(directory/f"case{index}.{arm}_quality.json", evidence)
        empty_ep_binding(phase, qual)
        verify_reference(pair["quality"][arm], directory/f"case{index}.{arm}_quality.json", phase, evidence)
        require(phase.get("name") == f"case{index}.{arm}_quality" and phase.get("repeat") is None,
                "quality phase filename/repeat mismatch")
        masks[arm] = audit_phase(phase, row, preflight, arm=arm, timing=False, quality=True)
        fresh_generation(phase)
        rec = phase["record"]
        quality[arm] = {key: rec[key] for key in ("text", "answer_em", "answer_token_f1", "truncated", "finish_reason")}
        quality[arm]["generated_tokens"] = len(rec["output_ids"])
        warmup = read(directory/f"case{index}.{arm}_warmup.json", evidence)
        empty_ep_binding(warmup, qual)
        require(warmup.get("name") == f"case{index}.{arm}_warmup" and warmup.get("repeat") is None,
                "warmup phase filename/repeat mismatch")
        audit_phase(warmup, row, preflight, arm=arm, timing=False, quality=False)
        fresh_generation(warmup)
    expected = [(repeat, ARMS[(index+repeat+slot)%len(ARMS)]) for repeat in range(3) for slot in range(len(ARMS))]
    phases = pair.get("timing_phases")
    require(type(phases) is list and [(p.get("repeat"), p.get("arm")) for p in phases] == expected,
            "three rotating same-case paired C1 rounds required")
    for recorded, (repeat, arm) in zip(phases, expected):
        phase = read(directory/f"case{index}.{arm}_measure{repeat}.json", evidence)
        empty_ep_binding(phase, qual)
        verify_reference(recorded, directory/f"case{index}.{arm}_measure{repeat}.json", phase, evidence,
                         extras=("repeat","arm"))
        require(phase.get("name") == f"case{index}.{arm}_measure{repeat}"
                and phase.get("repeat") == repeat and phase.get("arm") == arm, "formal timing schedule mismatch")
        audit_phase(phase, row, preflight, arm=arm, timing=True, quality=False)
        fresh_generation(phase)
        timing[arm].append(phase["record"]["online_ttft_s"])
    builds = []
    for doc_index, (_, _, doc_key) in enumerate(documents(row)):
        phase = read(directory/f"case{index}.offline_doc{doc_index}.json", evidence)
        empty_ep_binding(phase, qual)
        binding(phase, preflight)
        require(phase.get("arm") == "offline" and phase.get("mode") == "capture_independent"
                and phase.get("input_ids_sha256") == doc_key, "offline document phase mismatch")
        require(phase["controller"].get("offline_yoco_full_layout") is True
                and phase["controller"].get("model_boundary_restored") is True,
                "offline all-layer capture must restore native model YOCO before online runs")
        audit_native_no_cache(phase["controller"])
        payloads = phase["controller"].get("offline_payloads")
        require(type(payloads) is list and len(payloads) == len(CAPTURE_LAYERS), "all offline layers required")
        by_layer = {item["scope"]["layer"]: item for item in payloads}
        require(set(by_layer) == set(CAPTURE_LAYERS), "offline layer coverage changed")
        for layer, item in by_layer.items():
            name = doc_key+f".layer{layer}.tail128.bf16"
            scope = dict(model_sha256=preflight["identity"]["checkpoint_index_sha256"],
                tokenizer_sha256=preflight["identity"]["tokenizer_json_sha256"],
                execution_sha256=preflight["identity"]["execution_configuration_sha256"],
                document_sha256=doc_key, layer=layer, representation="swa-tail128-postnorm-prerope-bf16",
                source_local_start=32640, source_document_tokens=32768, tail_tokens=128)
            require(item.get("scope") == scope and item.get("file") == name
                    and item.get("shape") == [128, 512] and item.get("dtype") == "bfloat16"
                    and item.get("bytes") == 128*512*2, "offline KV identity/geometry mismatch")
            path = directory/f"case{index}.offline-kv"/name
            require(path.parent.is_dir() and not path.parent.is_symlink(), "tail bank directory symlink rejected")
            evidence[f"case{index}.offline-kv/"+name] = audit_bank_payload(path, item["sha256"])
        builds.append(phase["sha256"])
    require(builds == pair.get("offline_phase_sha256"), "offline construction closure mismatch")
    medians = {arm: statistics.median(samples) for arm, samples in timing.items()}
    baseline = medians["recomputed"]
    comparisons = {arm: dict(em_loss_pp=100*(quality["recomputed"]["answer_em"]-quality[arm]["answer_em"]),
        f1_loss_pp=100*(quality["recomputed"]["answer_token_f1"]-quality[arm]["answer_token_f1"]),
        ttft_reduction_s=baseline-medians[arm], ttft_reduction_percent=100*(1-medians[arm]/baseline),
        ttft_speedup=baseline/medians[arm]) for arm in ARMS[1:]}
    return pair, dict(index=index, dataset=row["dataset"], references=row["references"], question=row["question"],
        input_tokens=row["input_tokens"], quality=quality, online_ttft_samples_s=timing,
        online_ttft_p50_s=medians, actual_rows_and_swa_reuse=masks, comparisons=comparisons,
        offline_build_s=number(pair.get("offline_build_s"), "offline build seconds"), closure_sha256=pair["sha256"])


def summarize_cases(cases):
    """Same closed cases only; two empirical points, never a global frontier."""
    if not cases:
        return None
    arms = {}
    for arm in ARMS:
        samples = [sample for case in cases for sample in case["online_ttft_samples_s"][arm]]
        arms[arm] = dict(n=len(cases),
            em_percent=100*sum(case["quality"][arm]["answer_em"] for case in cases)/len(cases),
            f1_percent=100*sum(case["quality"][arm]["answer_token_f1"] for case in cases)/len(cases),
            online_ttft_p50_s=statistics.median(samples), online_ttft_samples_s=samples,
            truncated_count=sum(case["quality"][arm]["truncated"] for case in cases))
    baseline = arms["recomputed"]
    comparisons, rates_by_arm = {}, {}
    for arm in ARMS[1:]:
        candidate = arms[arm]
        totals={name:sum(case["actual_rows_and_swa_reuse"][arm].get(name,0) for case in cases)
                for name in ("swa_full_rows","swa_online_rows","swa_reused_rows","ffn_full_rows",
                             "ffn_selected_rows","attention_full_rows","attention_selected_rows",
                             "encoder_rows","encoder_selected_rows","decoder_rows","decoder_selected_rows",
                             "materialized_memory_cached_tail_rows","materialized_memory_online_non_document_rows",
                             "consumed_memory_cached_tail_pairs","consumed_memory_online_non_document_pairs",
                             "materialized_memory_online_document_rows","consumed_memory_online_document_pairs",
                             "dynamic_refreshed_rows","dynamic_fallback_memory_rows","dynamic_tail_slot_capacity",
                             "dynamic_unrefreshed_yoco_tail_slots","dynamic_seed_tail_slots","h2d_bytes")}
        materialized = sum(totals[k] for k in ("materialized_memory_cached_tail_rows",
            "materialized_memory_online_non_document_rows", "materialized_memory_online_document_rows"))
        consumed = sum(totals[k] for k in ("consumed_memory_cached_tail_pairs",
            "consumed_memory_online_non_document_pairs", "consumed_memory_online_document_pairs"))
        rates_by_arm[arm]=dict(totals,
            dynamic_refreshed_tail_slot_fraction=totals["dynamic_refreshed_rows"]/totals["dynamic_tail_slot_capacity"] if totals["dynamic_tail_slot_capacity"] else None,
            dynamic_tail_slot_denominator="all eight document tail128 slots for each layer4..39, including YOCO-removed positions",
            swa_reuse_fraction=totals["swa_reused_rows"]/totals["swa_full_rows"] if totals["swa_full_rows"] else 0.,
            ffn_selected_fraction=totals["ffn_selected_rows"]/totals["ffn_full_rows"] if totals["ffn_full_rows"] else 0.,
            attention_selected_fraction=totals["attention_selected_rows"]/totals["attention_full_rows"] if totals["attention_full_rows"] else 0.,
            compute_denominator="sum quality-phase native/YOCO surviving prefill rows in layers3..39; first3 excluded",
            swa_reuse_denominator="actual current document-tail rows plus online non-document rows; skipped middle rows excluded",
            materialized_memory_reuse_fraction=totals["materialized_memory_cached_tail_rows"]/materialized if materialized else 0.,
            materialized_memory_denominator="distinct SWA memory entries materialized per layer/chunk; repeated across chunks, not query-key pairs consumed",
            consumed_memory_reuse_fraction=totals["consumed_memory_cached_tail_pairs"]/consumed if consumed else 0.,
            consumed_memory_denominator="SWA query-key pairs actually consumed by selected query rows in quality phases; global KV excluded",
            not_model_flops_fraction=True,baseline_zero_counters_mean_controller_absent=True)
        comparisons[arm]=dict(em_loss_pp=baseline["em_percent"]-candidate["em_percent"],
            f1_loss_pp=baseline["f1_percent"]-candidate["f1_percent"],
            ttft_reduction_s=baseline["online_ttft_p50_s"]-candidate["online_ttft_p50_s"],
            ttft_reduction_percent=100*(1-candidate["online_ttft_p50_s"]/baseline["online_ttft_p50_s"]),
            ttft_speedup=baseline["online_ttft_p50_s"]/candidate["online_ttft_p50_s"],
            correct_to_wrong=sum(case["quality"]["recomputed"]["answer_em"] == 1 and
                                 case["quality"][arm]["answer_em"] == 0 for case in cases),
            wrong_to_correct=sum(case["quality"]["recomputed"]["answer_em"] == 0 and
                                 case["quality"][arm]["answer_em"] == 1 for case in cases))
    frontiers = {}
    for metric in ("em_percent", "f1_percent"):
        def dominates(a, b):
            return (arms[a][metric] >= arms[b][metric]
                    and arms[a]["online_ttft_p50_s"] <= arms[b]["online_ttft_p50_s"]
                    and (arms[a][metric] > arms[b][metric]
                         or arms[a]["online_ttft_p50_s"] < arms[b]["online_ttft_p50_s"]))
        frontiers[metric+"_vs_ttft"] = [a for a in ARMS if not any(dominates(b,a) for b in ARMS if a != b)]
    return dict(n=len(cases), arms=arms, comparison=comparisons["tail128_link"],
                comparisons=comparisons, actual_rates=rates_by_arm["tail128_link"],
                actual_rates_by_arm=rates_by_arm, empirical_frontiers=frontiers,
                point_set=list(ARMS), global_frontier_claim=False, statistical_significance_claim=False)


def validate_closed_run(run, preflight, qual, smoke, closures, errors, directory, evidence, dynamic_smoke=None):
    sealed(run)
    require(not errors and len(closures) == 4 and qual is not None and smoke is not None
            and run.get("schema") == RUN_SCHEMA and run.get("status") == "complete"
            and run.get("completed_case_indices") == list(CASE_INDICES)
            and [c.get("index") for c in closures] == list(CASE_INDICES)
            and run.get("identity") == preflight["identity"]
            and run.get("manifest_sha256") == preflight["manifest_sha256"]
            and run.get("qualification") == qual and run.get("partial_smoke_sha256") == smoke["sha256"]
            and dynamic_smoke is not None and run.get("dynamic_partial_smoke_sha256") == dynamic_smoke["sha256"]
            and type(run.get("cases")) is list and len(run["cases"]) == 4,
            "final run does not close all four independently audited cases")
    for ref, closure in zip(run["cases"], closures):
        require(ref.get("index") == closure["index"], "final case order mismatch")
        verify_reference(ref, Path(directory)/f"case{closure['index']}.paired.json", closure,
                         evidence, extras=("index",))
    return True


def audit(directory, case_index=None):
    """Audit new closed cases with all per-case forty-MiB offline tail banks.

    ``case_index`` deliberately emits a SUBSET result and never a final complete
    run claim, even if run.json already exists. The final audit without this
    option still checks every case, every bank and the complete run closure.
    """
    directory = Path(directory)
    require(case_index is None or type(case_index) is int and case_index in CASE_INDICES,
            "requested subset case is not registered")
    evidence, errors, results, closures = {}, [], [], []
    preflight = read(directory/"preflight.json", evidence)
    require(preflight.get("schema") == RUN_SCHEMA and preflight.get("protocol") == tail_link_protocol()
            and preflight["protocol"].get("cases") == list(CASE_INDICES)
            and preflight.get("evaluation_case_indices") == list(CASE_INDICES), "foreign preflight/protocol")
    audit_native_swa_configuration(preflight)
    source_before = frozen_sources(preflight)
    original = read(directory/"input_manifest.json", evidence)
    require(original["sha256"] == preflight["manifest_sha256"] == preflight["protocol"]["input_manifest_sha256"],
            "original fixed input/gold seal differs")
    original_by_index = {r["index"]: r for r in original["cohort"]}
    rows = preflight.get("audit_cases")
    require(type(rows) is list and [r.get("index") for r in rows] == list(CASE_INDICES)
            and [r.get("dataset") for r in rows] == ["hotpotqa", "2wikimqa"]*2,
            "fixed four-case cohort required; no score-based case selection")
    for row in rows:
        require(type(row.get("input_tokens")) is int and row["input_tokens"] >= 262145, "8x32K input required")
        validate_digest(row.get("input_ids_sha256"), "case input")
        documents(row)
        source = original_by_index[row["index"]]
        require(all(row[k] == source[k] for k in row), "audit case differs from original frozen input/gold")
        ids = source["input_ids"]
        require(type(ids) is list and len(ids) == row["input_tokens"]
                and all(type(x) is int and x >= 0 for x in ids)
                and digest(ids) == row["input_ids_sha256"], "original raw input token hash differs")
        for lo, hi, key in documents(row):
            require(digest(ids[lo:hi]) == key, "original document span/token hash differs")
    qual, smoke, smoke_summary = None, None, None
    dynamic_smoke = dynamic_smoke_summary = None
    if (directory/"qualification.json").exists():
        qual = qualification(directory, preflight, evidence)
    if (directory/"partial_smoke.json").exists():
        require(qual is not None, "partial smoke exists without strict native qualification")
        smoke, smoke_summary = partial_smoke(directory, preflight, qual, evidence)
    if (directory/"dynamic_partial_smoke.json").exists():
        require(qual is not None, "dynamic smoke exists without strict native qualification")
        dynamic_smoke, dynamic_smoke_summary = partial_smoke(directory, preflight, qual, evidence, arm="tail128_dynamic")
    if smoke_summary is not None and dynamic_smoke_summary is not None:
        require(smoke_summary["first_three_fingerprints_sha256"] == dynamic_smoke_summary["first_three_fingerprints_sha256"],
                "dynamic policy changed first-three layer outputs/mHC relative to static on full input")
    requested = rows if case_index is None else [r for r in rows if r["index"] == case_index]
    if any((directory/f"case{r['index']}.paired.json").exists() for r in requested):
        require(qual is not None and smoke is not None and dynamic_smoke is not None, "formal case exists before both qualified partial smokes")
    if smoke is not None and dynamic_smoke is not None:
        for row in requested:
            if not (directory/f"case{row['index']}.paired.json").exists():
                continue
            try:
                closure, result = audit_case(directory, row, preflight, qual, smoke, evidence, dynamic_smoke)
                closures.append(closure)
                results.append(result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(dict(index=row["index"], error=str(exc)))
    complete = False
    if case_index is None and (directory/"run.json").exists():
        run = read(directory/"run.json", evidence)
        complete = validate_closed_run(run, preflight, qual, smoke, closures, errors, directory, evidence, dynamic_smoke)
    subset_complete = case_index is not None and not errors and len(results) == 1
    status = "invalid" if errors else "complete" if complete else "subset_complete" if subset_complete else "incomplete"
    require(frozen_sources(preflight) == source_before, "frozen sources changed during CPU audit")
    output = dict(schema=SCHEMA, status=status, frozen_sources_verified=source_before,
        completed_case_indices=[item["index"] for item in results], complete_cases=len(results), planned_cases=4,
        audit_scope="all_registered_cases" if case_index is None else "explicit_single_case_subset",
        requested_case_indices=[r["index"] for r in requested], final_run_audited=complete,
        overall_run_complete=complete, all_registered_cases_audited=complete,
        qualification_passed=qual is not None, partial_smoke=smoke_summary,
        dynamic_partial_smoke=dynamic_smoke_summary,
        cases=results, errors=errors, evidence=evidence,
        summary=summarize_cases(results),
        by_dataset={name:summarize_cases([c for c in results if c["dataset"]==name])
                    for name in ("hotpotqa", "2wikimqa")},
        manifest_sha256=preflight["manifest_sha256"], execution_configuration_sha256=preflight["identity"]["execution_configuration_sha256"],
        protocol_sha256=preflight["protocol"]["sha256"], cuda_initialized=False,
        exploratory_previously_exposed_inputs=True, statistical_generalization=False,
        cold_start_claim=False, no_flops_fraction_claim=True, c4_measured=False)
    return dict(output, sha256=digest(output))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-index", type=int, choices=CASE_INDICES,
                        help="independent single-case subset; never a final run-complete claim")
    args = parser.parse_args(argv)
    if sys.platform.startswith("linux"):
        import resource
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        bound=96*1024**3
        if soft != resource.RLIM_INFINITY:bound=min(bound,soft)
        if hard != resource.RLIM_INFINITY:bound=min(bound,hard)
        resource.setrlimit(resource.RLIMIT_AS,(bound,hard))
    result = audit(args.result_dir, case_index=args.case_index)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload, end="")
    return 0 if result["status"] in ("complete", "subset_complete") and not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

