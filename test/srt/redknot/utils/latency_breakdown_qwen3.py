#!/usr/bin/env python3
"""Qwen3-32B latency breakdown: baseline vs RedKnot prefill/decode."""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
import torch
from sglang.srt.layers.attention.redknot import (
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_redknot_offlinekv,
)

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH", "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B"
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASET = os.environ.get("REDKNOT_DATASETS", "triviaqa")
CTXS = [int(x) for x in os.environ.get("REDKNOT_CTXS", "8192,16384,32768").split(",")]
N = int(os.environ.get("REDKNOT_N_SAMPLES", "3"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "32"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK", "4000"))
HEAD_CFG = (
    Path(__file__).resolve().parent / "head_class/qwen3-32B_optimal_g15_lf_ret.json"
)
FFN_CFG = Path(__file__).resolve().parent / "sparse_ffn_params/qwen3-32B.json"
OUT = os.environ.get(
    "REDKNOT_LATENCY_OUT",
    str(Path(__file__).resolve().parent / "figures/latency_qwen3_32b.json"),
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_baseline(model, tok, ids):
    _sync()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    nxt = out.logits[0, -1, :].argmax().view(1, 1)
    _sync()
    t_pf = time.perf_counter() - t0
    past, g = out.past_key_values, [int(nxt[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        og = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = og.past_key_values, og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        g.append(tid)
        if tid == tok.eos_token_id:
            break
    _sync()
    t_dec = max(time.perf_counter() - t1, 1e-6)
    return dict(n_in=ids.shape[1], n_out=len(g), t_prefill=t_pf, t_dec=t_dec)


@torch.no_grad()
def run_redknot(model, tok, segs, query):
    _sync()
    t0 = time.perf_counter()
    from transformers import DynamicCache, TextStreamer

    cache = DynamicCache(config=model.config)
    pos = 0
    last = None
    for seg in segs:
        ids = tok(seg, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        pids = torch.arange(pos, pos + ids.shape[1], device=model.device).unsqueeze(0)
        out = model(
            input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
        )
        cache, last = out.past_key_values, out.logits[0, -1, :]
        pos += ids.shape[1]
    ids_q = tok(query, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    pids_q = torch.arange(pos, pos + ids_q.shape[1], device=model.device).unsqueeze(0)
    out = model(
        input_ids=ids_q, position_ids=pids_q, past_key_values=cache, use_cache=True
    )
    cache, last = out.past_key_values, out.logits[0, -1, :]
    pos += ids_q.shape[1]
    nxt = last.argmax().view(1, 1)
    _sync()
    t_pf = time.perf_counter() - t0
    g = [int(nxt[0, 0])]
    t1 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        pids = torch.tensor([[pos]], device=model.device)
        og = model(
            input_ids=nxt, position_ids=pids, past_key_values=cache, use_cache=True
        )
        cache, nxt = og.past_key_values, og.logits[0, -1, :].argmax().view(1, 1)
        tid = int(nxt[0, 0])
        g.append(tid)
        pos += 1
        if tid == tok.eos_token_id:
            break
    _sync()
    t_dec = max(time.perf_counter() - t1, 1e-6)
    total_tokens = (
        sum(tok(seg, add_special_tokens=False)["input_ids"].__len__() for seg in segs)
        + ids_q.shape[1]
    )
    return dict(n_in=total_tokens, n_out=len(g), t_prefill=t_pf, t_dec=t_dec)


def load_samples(tok, n_segs, n):
    raw = []
    path = os.path.join(LB, f"{DATASET}.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input") and r.get("context"):
                raw.append(r)
    import random

    random.Random(0).shuffle(raw)
    out = []
    for i in range(min(n, len(raw))):
        ctx_toks = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % len(raw)
        while len(ctx_toks) < CHUNK * n_segs and j != i:
            ctx_toks += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % len(raw)
        ctx_toks = ctx_toks[: CHUNK * n_segs]
        segs = [
            tok.decode(ctx_toks[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, len(ctx_toks), CHUNK)
            if ctx_toks[k : k + CHUNK]
        ]
        out.append(dict(q=raw[i]["input"], segs=segs))
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hc = HeadClassConfig.from_json(str(HEAD_CFG))
    hc.merge_retrieval_to_global()
    with open(FFN_CFG) as f:
        ffn_cfg = json.load(f)
    print(
        f"Model: {MODEL} | ctxs={CTXS} | N={N} | head global={hc.summary().get('global')}/{hc.summary()['total']}"
    )
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    print(f"model loaded. running latency breakdown...")

    results = {}
    for ctx in CTXS:
        n_segs = ctx // CHUNK
        samples = load_samples(tok, n_segs, N)
        if not samples:
            print(f"skip {ctx}")
            continue
        b_rows, r_rows = [], []
        for s in samples:
            full = "\n\n".join(s["segs"]) + "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ids = tok(full, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(model.device)
            b = run_baseline(model, tok, ids)
            b_rows.append(b)
            print(
                f"  [base ctx={ctx}] pf={b['t_prefill'] * 1000:.0f}ms dec={b['t_dec'] * 1000:.0f}ms"
            )
        for s in samples:
            _sync()
            t0 = time.perf_counter()
            first_logits, text, query_len, ttft = run_redknot_offlinekv(
                model,
                tok,
                segments_offline=offline_prefill_segments(model, tok, s["segs"]),
                query_text="\n\nQuestion: " + s["q"] + "\nAnswer:",
                head_cfg=hc,
                sparse_ffn_schedule=SparseFFNSchedule(**ffn_cfg),
                max_new_tokens=MAX_NEW,
            )
            _sync()
            t_total = time.perf_counter() - t0
            t_dec = t_total - ttft
            n_tokens = (
                len(
                    tok.decode(
                        first_logits.argmax(dim=-1).tolist(), skip_special_tokens=True
                    ).split()
                )
                + 1
            )
            r = dict(
                t_prefill=ttft,
                t_dec=t_dec,
                n_in=sum(
                    len(tok(seg, add_special_tokens=False)["input_ids"])
                    for seg in s["segs"]
                ),
                n_out=query_len + MAX_NEW,
            )
            r_rows.append(r)
            print(f"  [rk   ctx={ctx}] pf={ttft * 1000:.0f}ms dec={t_dec * 1000:.0f}ms")
        base_avg = {k: sum(r[k] for r in b_rows) / len(b_rows) for k in b_rows[0]}
        rk_avg = {k: sum(r[k] for r in r_rows) / len(r_rows) for k in r_rows[0]}
        results[str(ctx)] = {"base": base_avg, "rk": rk_avg}
        print(
            f"  ctx={ctx}: base pf={base_avg['t_prefill'] * 1000:.0f}ms rk pf={rk_avg['t_prefill'] * 1000:.0f}ms speedup={base_avg['t_prefill'] / max(rk_avg['t_prefill'], 1e-9):.2f}x"
        )

    print(f"\n===== SUMMARY =====")
    print(f"{'ctx':>8} | {'base pf ms':>10} {'rk pf ms':>9} | {'pf speedup':>9}")
    for ctx in CTXS:
        r = results[str(ctx)]
        spd = r["base"]["t_prefill"] / max(r["rk"]["t_prefill"], 1e-9)
        print(
            f"{ctx:>8} | {r['base']['t_prefill'] * 1000:>10.0f} {r['rk']['t_prefill'] * 1000:>9.0f} | {spd:>8.2f}x"
        )

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
