#!/usr/bin/env python3
"""Test RAG v2: OFFLINE prefix status + ONLINE window recompute, vs standard.

Offline: build doc state (global=full status, local=prefix status at doc_len-W,
full KV cached). Online: query reuses prefix, local heads recompute window only.
Verify F1/EM vs standard full recompute, and report online token count (should
be ~Wmax+query, not full doc -> the saving).
"""

from __future__ import annotations
import json, os, random, re, string, sys, time
from collections import Counter
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = os.environ.get(
    "REDKNOT_DATASETS", "triviaqa,hotpotqa,2wikimqa,multifieldqa_en"
).split(",")
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
N_CHUNK = int(os.environ.get("REDKNOT_N_CHUNK", "10"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "24"))
DENSE_PREFIX = 5
DECAY_Q = 0.95
SAFETY = 4.0
MINW = 512
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
USE_COMPILE = os.environ.get("REDKNOT_COMPILE", "0") == "1"
FORCE_WINDOW = int(os.environ.get("REDKNOT_FORCE_WINDOW", "0"))
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "0"))
MOE_SPARSE = os.environ.get("REDKNOT_MOE_SPARSE", "0") == "1"
DEEP_MOE_START = int(os.environ.get("REDKNOT_DEEP_MOE_START", "20"))
MOE_MASS_THRESH = float(os.environ.get("REDKNOT_MOE_MASS_THRESH", "0.2"))
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def _n(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, gs):
    b = 0.0
    for g in gs:
        a, c = _n(p).split(), _n(g).split()
        if not a or not c:
            b = max(b, float(a == c))
            continue
        cm = Counter(a) & Counter(c)
        ns = sum(cm.values())
        if ns == 0:
            continue
        pr, rc = ns / len(a), ns / len(c)
        b = max(b, 2 * pr * rc / (pr + rc))
    return b


def em(p, gs):
    return max((float(_n(p) == _n(g)) for g in gs), default=0.0)


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def load(name, tok, n, tgt_tokens=None):
    raw = [
        json.loads(l)
        for l in open(os.path.join(LB, f"{name}.jsonl"))
        if json.loads(l).get("input")
        and json.loads(l).get("context")
        and json.loads(l).get("answers")
    ]
    random.Random(0).shuffle(raw)
    tgt = tgt_tokens if tgt_tokens is not None else N_CHUNK * CHUNK
    out = []
    nr = len(raw)
    for i in range(nr):
        if len(out) >= n:
            break
        tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = (i + 1) % nr
        while len(tk) < tgt and j != i:
            tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j = (j + 1) % nr
        tk = tk[:tgt]
        if len(tk) < tgt:
            continue
        ch = [
            tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
            for k in range(0, tgt, CHUNK)
        ]
        out.append(
            {"q": raw[i]["input"], "golds": raw[i]["answers"], "chunks": ch, "ds": name}
        )
    return out


@torch.no_grad()
def std(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    o = model(input_ids=ids, use_cache=True)
    nx = o.logits[0, -1, :].argmax().view(1, 1)
    p = o.past_key_values
    g = [int(nx[0, 0])]
    for _ in range(MAX_NEW - 1):
        og = model(input_ids=nx, past_key_values=p, use_cache=True)
        p = og.past_key_values
        nx = og.logits[0, -1, :].argmax().view(1, 1)
        t = int(nx[0, 0])
        g.append(t)
        if t == tok.eos_token_id:
            break
    return tok.decode(g, skip_special_tokens=True)


@torch.no_grad()
def std_prefill_ttft(model, tok, text):
    """Standard dense baseline: measure pure PREFILL TTFT (time to first token)
    over the full docs+query — the fair, hard-to-beat dense baseline that
    RedKnot's online window-reuse path is compared against. Also completes the
    generation so the FULL output text can be compared by eye.

    Returns (full_text, ttft_seconds)."""
    import time as _t

    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    seqlen = ids.shape[1]
    # Chunked dense prefill: feed the full context in slices through the KV cache
    # so activation memory stays bounded (a single 64K forward OOMs on a 2-GPU
    # device_map). This is mathematically the SAME dense attention; we just bound
    # the per-step activation. TTFT = wall time of the whole dense prefill.
    PREFILL_CHUNK = int(os.environ.get("REDKNOT_BASELINE_PREFILL_CHUNK", "8000"))
    from transformers import DynamicCache

    torch.cuda.synchronize()
    t0 = _t.perf_counter()
    cache = DynamicCache(config=model.config)
    pos = 0
    o = None
    for st in range(0, seqlen, PREFILL_CHUNK):
        piece = ids[:, st : st + PREFILL_CHUNK]
        pids = torch.arange(pos, pos + piece.shape[1], device=ids.device).unsqueeze(0)
        o = model(
            input_ids=piece,
            position_ids=pids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = o.past_key_values
        pos += piece.shape[1]
    nx = o.logits[0, -1, :].argmax().view(1, 1)
    torch.cuda.synchronize()
    ttft = _t.perf_counter() - t0
    # Complete the generation (outside the TTFT measurement) for text comparison.
    p = cache
    g = [int(nx[0, 0])]
    for _ in range(MAX_NEW - 1):
        pids = torch.tensor([[pos]], device=ids.device)
        og = model(
            input_ids=nx,
            position_ids=pids,
            past_key_values=p,
            use_cache=True,
            logits_to_keep=1,
        )
        p = og.past_key_values
        nx = og.logits[0, -1, :].argmax().view(1, 1)
        t = int(nx[0, 0])
        g.append(t)
        pos += 1
        if t == tok.eos_token_id:
            break
    return tok.decode(g, skip_special_tokens=True), ttft


@torch.no_grad()
def build_win(model, tok, text):
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    bm = model.model

    # Fast path: a fixed forced window does NOT need the decay-measurement forward
    # (which runs a full-context pass and OOMs at long ctx). Build the window map
    # directly from the layer's head count.
    if FORCE_WINDOW > 0:
        win = {}
        for li in linear_attention_layer_indices(model.config):
            if li < DENSE_PREFIX:
                win[li] = None
                continue
            n_heads = bm.layers[li].linear_attn.num_v_heads
            win[li] = torch.full(
                (n_heads,), FORCE_WINDOW, dtype=torch.long, device=model.device
            )
        return win

    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        model.device
    )
    hs = {}
    hh = []
    for li in linear_attention_layer_indices(model.config):

        def mk(_li):
            def hook(m, a, k):
                h = a[0] if a and torch.is_tensor(a[0]) else k.get("hidden_states")
                if h is not None:
                    hs[_li] = h.detach()

            return hook

        hh.append(
            bm.layers[li].linear_attn.register_forward_pre_hook(
                mk(li), with_kwargs=True
            )
        )
    model(input_ids=ids, use_cache=False)
    for h in hh:
        h.remove()
    decay = measure_linear_head_decay(model, hs, decay_quantile=DECAY_Q)
    ctx = N_CHUNK * CHUNK
    win = {}
    for li, d in decay.items():
        if li < DENSE_PREFIX:
            win[li] = None
            continue
        if FORCE_WINDOW > 0:
            win[li] = torch.full_like(d, FORCE_WINDOW, dtype=torch.long)
            continue
        ml = 1.0 / (1.0 - d.clamp(max=0.99999))
        wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
        wt = torch.where(wt >= ctx, torch.zeros_like(wt), wt)
        if WINDOW_CAP > 0:
            # Promote oversized local heads to GLOBAL (window=0 -> reuse S_T,
            # exact, zero online cost) so Wmax stays small -> short online seq.
            wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
        win[li] = wt
    return win


@torch.no_grad()
def _run_one_length(
    model, tok, ctx_tokens, rag_build_offline_v2, rag_query_reuse_v2, warmed
):
    """Score all datasets at one context length. Returns (sumF1_std, sumF1_v2,
    sumTTFT_std, sumTTFT_v2, n)."""
    W = 92
    samples = []
    for ds in DATASETS:
        samples += load(ds, tok, N, tgt_tokens=ctx_tokens)
    print("=" * W)
    print(
        f" CONTEXT = {ctx_tokens:,} tokens | datasets={DATASETS} | "
        f"compile={USE_COMPILE} window={FORCE_WINDOW or 'adaptive'}"
    )
    print("=" * W)
    if not samples:
        print(" (no samples at this length)")
        return 0.0, 0.0, 0.0, 0.0, 0

    import time as _t

    # Warm up at THIS length (compile graph + autotune) so per-sample TTFT is steady.
    if not warmed["done"] or True:
        _w = samples[0]
        _full = "\n\n".join(_w["chunks"]) + QP.format(q=_w["q"])
        try:
            std_prefill_ttft(model, tok, _full)
            _win = build_win(model, tok, _full)
            _doc = rag_build_offline_v2(
                model, tok, segments=_w["chunks"], win_tok_by_layer=_win
            )
            rag_query_reuse_v2(
                model,
                tok,
                doc_state=_doc,
                query_text=QP.format(q=_w["q"]),
                max_new_tokens=2,
                use_compile=USE_COMPILE,
                moe_sparse=MOE_SPARSE,
                deep_moe_start_layer=DEEP_MOE_START,
                moe_mass_thresh=MOE_MASS_THRESH,
            )
            warmed["done"] = True
        except Exception as _e:  # noqa: BLE001
            print(f" [warmup skipped: {_e}]")

    sf = rf = se = re_ = bt = rt = 0.0
    n = 0

    def _clean(t):
        t = (t or "").strip()
        t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
        return " ".join(t.split())[:200]

    for s in samples:
        qt = QP.format(q=s["q"])
        full = "\n\n".join(s["chunks"]) + qt

        sb_raw, bttft = std_prefill_ttft(model, tok, full)
        sb = short(sb_raw)

        torch.cuda.synchronize()
        _tb = _t.perf_counter()
        win = build_win(model, tok, full)
        doc = rag_build_offline_v2(
            model, tok, segments=s["chunks"], win_tok_by_layer=win
        )
        torch.cuda.synchronize()
        build_t = _t.perf_counter() - _tb

        rk_raw, rttft = rag_query_reuse_v2(
            model,
            tok,
            doc_state=doc,
            query_text=qt,
            max_new_tokens=MAX_NEW,
            use_compile=USE_COMPILE,
            moe_sparse=MOE_SPARSE,
            deep_moe_start_layer=DEEP_MOE_START,
            moe_mass_thresh=MOE_MASS_THRESH,
        )
        rk = short(rk_raw)
        sF, rF = f1(sb, s["golds"]), f1(rk, s["golds"])
        sf += sF
        rf += rF
        se += em(sb, s["golds"])
        re_ += em(rk, s["golds"])
        bt += bttft
        rt += rttft
        n += 1
        sp = bttft / rttft if rttft > 0 else 0.0

        print("·" * W)
        print(f" [{s['ds']}] Q: {s['q'][:110]}")
        print(f"   gold        : {s['golds']}")
        print(f"   DENSE  (all): {_clean(sb_raw)!r}")
        print(f"   REDKNOT(win): {_clean(rk_raw)!r}")
        print(
            f"   F1 dense={sF:.2f} redknot={rF:.2f} | ttft dense={bttft:.2f}s "
            f"redknot={rttft:.2f}s speedup={sp:.2f}x | build={build_t:.1f}s"
        )

    k = max(n, 1)
    print("-" * W)
    print(
        f" @ctx={ctx_tokens:,}: AVG F1 dense={sf / k:.3f} redknot={rf / k:.3f} "
        f"dF1={rf / k - sf / k:+.3f} | TTFT dense={bt / k:.2f}s redknot={rt / k:.2f}s "
        f"speedup={(bt / k) / (rt / k) if rt > 0 else 0:.2f}x (n={k})"
    )
    return sf, rf, bt, rt, n


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        rag_build_offline_v2,
        rag_query_reuse_v2,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()

    # Context lengths to sweep (tokens). Default keeps the legacy single length.
    ctx_lens = [
        int(x)
        for x in os.environ.get("REDKNOT_CTX_LENS", str(N_CHUNK * CHUNK)).split(",")
        if x.strip()
    ]
    warmed = {"done": False}
    grand = []
    for ctx in ctx_lens:
        sf, rf, bt, rt, n = _run_one_length(
            model, tok, ctx, rag_build_offline_v2, rag_query_reuse_v2, warmed
        )
        if n:
            grand.append((ctx, sf / n, rf / n, bt / n, rt / n, n))

    print("\n" + "=" * 92)
    print(" SUMMARY across context lengths")
    print("=" * 92)
    print(
        f" {'ctx':>8} {'n':>3} {'F1_dense':>9} {'F1_redknot':>11} "
        f"{'TTFT_dense':>11} {'TTFT_redknot':>13} {'speedup':>8}"
    )
    for ctx, fd, fr, td, tr, n in grand:
        print(
            f" {ctx:>8,} {n:>3} {fd:>9.3f} {fr:>11.3f} {td:>10.2f}s "
            f"{tr:>12.2f}s {(td / tr if tr > 0 else 0):>7.2f}x"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
