#!/usr/bin/env python3
"""HotpotQA 4x4K: full recompute baseline vs RedKnot (RedKnot + Sparse FFN).

Reads the local HotpotQA distractor parquet, builds 4-segment x 4K-token RAG
prompts, and compares:
  - Baseline: concatenate the 4 docs into one ~16K prompt and prefill fully.
  - RedKnot : per-doc offline prefill + position-independent online reuse with
              head-classified attention and token-selective Sparse FFN.

Reports F1 / EM / cosine / top-1 / top-10, TTFT, and analytical prefill FLOPs.
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import string
import sys
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.layers.attention.redknot import (  # noqa: E402
    HeadClassConfig,
    SparseFFNSchedule,
    offline_prefill_segments,
    run_baseline,
    run_redknot,
    run_redknot_batched,
    sparse_ffn_flops,
)

MODEL = "/workspace/096/models/Qwen3-32B"
HEAD_CFG = "/workspace/096/__REDKNOT_V02__/configs/qwen3-32B_w256_g15_lf_ret.json"
PARQUET = "/workspace/096/__REDKNOT_V02__/datasets/HotpotQA/distractor/validation-00000-of-00001.parquet"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── metrics ──
def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred, gold):
    return float(norm(pred) == norm(gold))


def short_ans(t):
    """Extract the answer span from the generation.

    The model emits the answer first, then often keeps "Question:/Answer:"
    self-chatting. The real answer is therefore the text BEFORE the first
    follow-on "Question:"/"Answer:" turn, with any reasoning block removed.
    """
    t = t or ""
    if not t.strip():
        return ""
    # Drop a <think>...</think> block if present.
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    # Cut at the first follow-on QA turn the model invents.
    t = re.split(r"\n\s*(?:question|q)\s*[:：]", t, flags=re.I)[0]
    t = re.split(r"\n\s*(?:answer|a)\s*[:：]", t, flags=re.I)[0]
    # If the head is an explicit "Answer:" marker, take what follows it.
    m = re.match(
        r"\s*(?:final answer|short answer|answer)\s*[:：]\s*(.+)", t, flags=re.I
    )
    if m:
        t = m.group(1)
    # First non-empty line is the answer span.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    cand = lines[0] if lines else t.strip()
    cand = cand.strip().strip('"').strip("'").strip()
    cand = re.sub(r"\s*[.。]\s*$", "", cand)
    if len(cand.split()) > 12:
        cand = re.split(r"[.;:]", cand)[0].strip()
    return cand


def flatten_ctx(row):
    ctx = row["context"]
    titles, sents = ctx["title"], ctx["sentences"]
    if hasattr(titles, "tolist"):
        titles = titles.tolist()
    if hasattr(sents, "tolist"):
        sents = sents.tolist()
    out = []
    for t, s in zip(titles, sents):
        if hasattr(s, "tolist"):
            s = s.tolist()
        out.append(f"{t}. {' '.join(map(str, s))}")
    return out


def sup_titles(row):
    sf = row["supporting_facts"]
    titles = sf["title"]
    if hasattr(titles, "tolist"):
        titles = titles.tolist()
    return set(map(str, titles))


def build_seg(parts, tok, target, filler):
    ids = tok("\n\n".join(parts), add_special_tokens=False)["input_ids"]
    fids = [
        tok("\n\n" + f, add_special_tokens=False)["input_ids"]
        for f in filler
        if f.strip()
    ] or [tok(" Continued.", add_special_tokens=False)["input_ids"]]
    i = 0
    while len(ids) < target:
        ids.extend(fids[i % len(fids)])
        i += 1
    return tok.decode(ids[:target], skip_special_tokens=True)


def build_samples(tok, n, n_seg, tps, seed):
    import random

    df = pd.read_parquet(PARQUET)
    idxs = list(range(len(df)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs:
        if len(out) >= n:
            break
        row = df.iloc[i]
        q, a = str(row["question"]), str(row["answer"])
        paras = flatten_ctx(row)
        gold_t = sup_titles(row)
        gold = [p for p in paras if p.split(".", 1)[0].strip() in gold_t]
        dist = [p for p in paras if p not in gold]
        if not gold:
            continue
        docs = [build_seg(gold + dist[:2], tok, tps, dist or paras)]
        filler = dist or paras
        for s in range(1, n_seg):
            docs.append(build_seg([filler[(s - 1) % len(filler)]], tok, tps, filler))
        out.append({"id": str(row["id"]), "question": q, "answer": a, "docs": docs})
    return out


def query_text(q):
    # Concise instruction; NO few-shot (it induces QA self-chatting). The
    # answer is taken as the first span the model emits.
    return (
        "\n\nUsing only the documents above, give the shortest exact answer "
        "span to the question (a name, entity, number, or short noun phrase). "
        "Answer with the span only, no explanation.\n"
        f"Question: {q}\nAnswer:"
    )


# ── FLOPs ──
def dims(cfg):
    qh = cfg.num_attention_heads
    kvh = cfg.num_key_value_heads
    h = cfg.hidden_size
    hd = getattr(cfg, "head_dim", h // qh)
    inter = getattr(cfg, "intermediate_size", h * 4)
    return cfg.num_hidden_layers, qh, kvh, hd, h, inter, qh // kvh


def baseline_flops(total, cfg):
    L, qh, kvh, hd, h, inter, _ = dims(cfg)
    attn = 4.0 * total * max(1, total // 2) * qh * hd
    ffn = 6.0 * total * h * inter
    return L * (attn + ffn)


def redknot_flops(doc_lens, qlen, hc, cfg):
    L, qh, kvh, hd, h, inter, qpk = dims(cfg)
    tot = 0.0
    prev = doc_lens[0] if doc_lens else 0
    for dl in doc_lens[1:]:
        for li in range(L):
            for kh in range(kvh):
                st = hc.get_strategy(li, kh)
                vis = (
                    min(prev, max(st.window, 0) + st.sink_size)
                    if st.is_local()
                    else prev
                )
                kvt = vis + max(1, dl // 2)
                tot += 4.0 * dl * kvt * qpk * hd
            tot += 6.0 * dl * h * inter
        prev += dl
    for li in range(L):
        tot += 4.0 * qlen * (prev + max(1, qlen // 2)) * qh * hd
        tot += 6.0 * qlen * h * inter
    return tot


def fmt(x):
    for u, s in (("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9)):
        if x >= s:
            return f"{x / s:.2f} {u}"
    return f"{x:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--n-segments", type=int, default=4)
    ap.add_argument("--tokens-per-segment", type=int, default=4000)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sparse-ffn", action="store_true")
    ap.add_argument("--kernel", default="fa2", choices=["fa2", "fa3"])
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Use batched (parallel) online prefill instead of serial.",
    )
    ap.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Segments per batched forward (0=all at once).",
    )
    ap.add_argument("--debug", action="store_true", help="Print raw generations.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"tokenizer {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"head config {HEAD_CFG}")
    hc = HeadClassConfig.from_json(HEAD_CFG)
    log(f"head summary {hc.summary()}")

    log("loading Qwen3-32B (device_map=auto, bf16, sdpa)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).eval()

    sched = None
    if args.sparse_ffn:
        sched = SparseFFNSchedule(dense_until=20, mass_thresh=0.5, recent_n=128)
        log(f"Sparse FFN on: {sched}")

    samples = build_samples(
        tok, args.n_samples, args.n_segments, args.tokens_per_segment, args.seed
    )
    log(
        f"built {len(samples)} samples (4x{args.tokens_per_segment} ~= {args.n_segments * args.tokens_per_segment} ctx)"
    )

    agg = collections.defaultdict(list)
    for si, s in enumerate(samples, 1):
        log("=" * 70)
        log(f"sample {si}/{len(samples)} id={s['id']}  Q={s['question'][:80]}")
        log(f"gold={s['answer']}")
        qt = query_text(s["question"])
        prompt = "\n\n".join(s["docs"]) + qt

        log("baseline full recompute ...")
        bl_logits, bl_text, bl_len, bl_ttft = run_baseline(
            model,
            tok,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            chunk_size=8192,
            attn_impl="sdpa",
        )
        bl_pred = short_ans(bl_text)

        log("RedKnot offline prefill per doc ...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        segs = offline_prefill_segments(
            model, tok, s["docs"], chunk_size=4096, model_id=MODEL
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        off_s = time.perf_counter() - t0

        mode = "PARALLEL (batched)" if args.parallel else "SERIAL"
        log(f"RedKnot online reuse [{mode}] ...")
        ffn_stats = []
        if args.parallel:
            mbs = args.micro_batch_size if args.micro_batch_size > 0 else None
            rc_logits, rc_text, qlen, rc_ttft = run_redknot_batched(
                model,
                tok,
                segments_offline=segs,
                query_text=qt,
                head_cfg=hc,
                max_new_tokens=args.max_new_tokens,
                kernel=args.kernel,
                micro_batch_size=mbs,
                sparse_ffn_schedule=sched,
                sparse_ffn_stats=ffn_stats if sched else None,
            )
        else:
            rc_logits, rc_text, qlen, rc_ttft = run_redknot(
                model,
                tok,
                segments_offline=segs,
                query_text=qt,
                head_cfg=hc,
                max_new_tokens=args.max_new_tokens,
                kernel=args.kernel,
                sparse_ffn_schedule=sched,
                sparse_ffn_stats=ffn_stats if sched else None,
            )
        rc_pred = short_ans(rc_text)

        if args.debug:
            print(f"  [raw baseline_text] {bl_text!r}")
            print(f"  [raw redknot_text ] {rc_text!r}")

        doc_lens = [sg.doc_len for sg in segs]
        bl_fl = baseline_flops(sum(doc_lens) + qlen, model.config)
        rc_fl = redknot_flops(doc_lens, qlen, hc, model.config)
        if sched and ffn_stats:
            deep = [x for x in ffn_stats if x.get("mode") == "sparse"]
            sf = sum(x["selected_frac"] for x in deep) / len(deep) if deep else 1.0
            L, qh, kvh, hd, h, inter, _ = dims(model.config)
            rep = sparse_ffn_flops(
                num_layers=L,
                tokens_per_layer=sum(doc_lens[1:]) or 1,
                hidden=h,
                intermediate=inter,
                schedule=sched,
                selected_frac_deep=sf,
            )
            log(
                f"  sparse-ffn deep_frac={sf:.3f} ffn_savings={rep['ffn_flops_savings'] * 100:.1f}%"
            )

        cos = torch.nn.functional.cosine_similarity(
            bl_logits.float(), rc_logits.float(), dim=-1
        ).item()
        top1 = int(bl_logits.argmax()) == int(rc_logits.argmax())
        top10 = (
            len(
                set(torch.topk(bl_logits, 10).indices.tolist())
                & set(torch.topk(rc_logits, 10).indices.tolist())
            )
            / 10
        )

        bf, rf = f1(bl_pred, s["answer"]), f1(rc_pred, s["answer"])
        be, re_ = em(bl_pred, s["answer"]), em(rc_pred, s["answer"])
        wall = bl_ttft / rc_ttft if rc_ttft > 0 else float("inf")
        fsp = bl_fl / rc_fl if rc_fl > 0 else float("inf")

        print(f"  baseline_pred={bl_pred!r}  redknot_pred={rc_pred!r}")
        print(
            f"  F1 bl={bf:.3f} rc={rf:.3f} | EM bl={be:.0f} rc={re_:.0f} | cos={cos:.4f} top1={top1} top10={top10:.0%}"
        )
        print(
            f"  TTFT baseline={bl_ttft:.2f}s redknot_online={rc_ttft:.2f}s (wall {wall:.2f}x, offline {off_s:.2f}s)"
        )
        print(
            f"  FLOPs baseline={fmt(bl_fl)} redknot={fmt(rc_fl)} (speedup {fsp:.2f}x, save {(1 - rc_fl / bl_fl) * 100:.1f}%)"
        )

        for k, v in (
            ("bf1", bf),
            ("rf1", rf),
            ("bem", be),
            ("rem", re_),
            ("cos", cos),
            ("top1", float(top1)),
            ("top10", top10),
            ("bttft", bl_ttft),
            ("rttft", rc_ttft),
            ("wall", wall),
            ("fsp", fsp),
            ("save", 1 - rc_fl / bl_fl),
        ):
            agg[k].append(v)
        del segs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean = lambda x: sum(x) / len(x) if x else 0.0
    print("\n" + "=" * 70)
    print(
        f"AGGREGATE over {len(samples)} samples | HotpotQA {args.n_segments}x{args.tokens_per_segment}"
    )
    print(f"  F1     baseline={mean(agg['bf1']):.3f}  redknot={mean(agg['rf1']):.3f}")
    print(f"  EM     baseline={mean(agg['bem']):.3f}  redknot={mean(agg['rem']):.3f}")
    print(
        f"  cosine={mean(agg['cos']):.4f}  top1={mean(agg['top1']):.0%}  top10={mean(agg['top10']):.0%}"
    )
    print(
        f"  TTFT   baseline={mean(agg['bttft']):.2f}s  redknot_online={mean(agg['rttft']):.2f}s  wall_speedup={mean(agg['wall']):.2f}x"
    )
    print(
        f"  FLOPs  speedup={mean(agg['fsp']):.2f}x  savings={mean(agg['save']) * 100:.1f}%"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
