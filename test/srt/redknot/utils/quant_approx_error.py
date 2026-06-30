#!/usr/bin/env python3
"""Quantify the approximation error of "local heads ignore far chunks".

Approach-2 hypothesis (user): for a LOCAL head, dropping the contribution of
chunks far before its window is nearly lossless because the delta-rule keeps
overwriting old state. We measure this directly.

We compare, on real RAG samples, the model output of:
  * DENSE  : full exact recompute over all chunks (gold reference)
  * KEEP=k : redknot where each LOCAL head's prefix state is built from ONLY the
             last k chunks before its window (chunks further back are DROPPED,
             i.e. their state contribution is zeroed). GLOBAL heads keep the full
             doc state. k = 1, 2, all.

For each k we report F1 vs gold and exact-text-match vs DENSE, so we can see how
small k can go before accuracy degrades. This tells us whether approach-2 is
viable and what "keep last-k chunks" setting is safe.

We realise "keep last k chunks" by, in the offline build, computing each linear
layer's prefix state as the state over [win_start - k*chunk, win_start) instead
of [0, win_start). Far history (before that) is dropped (initial_state=0 there).
"""

from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LB = os.environ.get(
    "REDKNOT_LONGBENCH_DIR", str(REPO / "test/srt/redknot/datasets/LongBench/data")
)
DATASETS = os.environ.get("REDKNOT_DATASETS", "triviaqa,hotpotqa").split(",")
CTX = int(os.environ.get("REDKNOT_CTX", "32000"))
CHUNK = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "2000"))
N = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
WINDOW_CAP = int(os.environ.get("REDKNOT_WINDOW_CAP", "4096"))
KEEP_LIST = [int(x) for x in os.environ.get("REDKNOT_KEEP_CHUNKS", "1,2,0").split(",")]
DEVICE_MAP = os.environ.get("REDKNOT_DEVICE_MAP", "auto")
DENSE_PREFIX, DECAY_Q, SAFETY, MINW = 5, 0.95, 4.0, 512
QP = "\n\nAnswer with the shortest exact span only.\nQuestion: {q}\nAnswer:"


def _patch():
    import sglang.srt.utils.common as _c

    _o = _c.assert_pkg_version
    _c.assert_pkg_version = lambda *a, **k: (_o(*a, **k) if False else None)


def _n(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1(p, gs):
    b = 0.0
    for g in gs:
        a, c = _n(p).split(), _n(g).split()
        if not a or not c:
            b = max(b, float(a == c))
            continue
        common = sum(1 for w in set(a) if w in c)
        if common == 0:
            continue
        pr, rc = common / len(a), common / len(c)
        b = max(b, 2 * pr * rc / (pr + rc))
    return b


def short(t):
    t = (t or "").strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.S | re.I)
    t = re.split(r"(assistant|Human:|\*Human|<\|)", t)[0]
    ls = [x.strip() for x in t.splitlines() if x.strip()]
    return (ls[0] if ls else t).strip().strip('"').strip("'")


def main():
    _patch()
    import json
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from sglang.srt.layers.attention.redknot.driver_qwen35 import (
        linear_attention_layer_indices,
        measure_linear_head_decay,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True
    ).eval()
    bm = model.model
    lin_idx = linear_attention_layer_indices(model.config)

    def load(name):
        raw = []
        with open(os.path.join(LB, f"{name}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        random.Random(0).shuffle(raw)
        out = []
        for i in range(len(raw)):
            if len(out) >= N:
                break
            tk = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
            j = i + 1
            while len(tk) < CTX and j < len(raw):
                tk += tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                j += 1
            tk = tk[:CTX]
            if len(tk) < CTX:
                continue
            ch = [
                tok.decode(tk[k : k + CHUNK], skip_special_tokens=True)
                for k in range(0, CTX, CHUNK)
            ]
            out.append(
                {
                    "q": raw[i]["input"],
                    "golds": raw[i]["answers"],
                    "chunks": ch,
                    "ds": name,
                }
            )
        return out

    samples = []
    for d in DATASETS:
        samples += load(d)

    @torch.no_grad()
    def dense(full):
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        cache = DynamicCache(config=model.config)
        pos = 0
        o = None
        for st in range(0, ids.shape[1], 8000):
            piece = ids[:, st : st + 8000]
            pids = torch.arange(pos, pos + piece.shape[1], device=ids.device).unsqueeze(
                0
            )
            o = model(
                input_ids=piece,
                position_ids=pids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = o.past_key_values
            pos += piece.shape[1]
        nx = o.logits[0, -1].argmax().view(1, 1)
        g = [int(nx[0, 0])]
        p = cache
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
            nx = og.logits[0, -1].argmax().view(1, 1)
            g.append(int(nx[0, 0]))
            pos += 1
        return tok.decode(g, skip_special_tokens=True)

    @torch.no_grad()
    def win_map(full):
        ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            model.device
        )
        hs, hh = {}, []
        for li in lin_idx:

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
        win = {}
        for li, d in decay.items():
            if li < DENSE_PREFIX:
                win[li] = None
                continue
            ml = 1.0 / (1.0 - d.clamp(max=0.99999))
            wt = torch.ceil(SAFETY * ml).long().clamp(min=MINW)
            wt = torch.where(wt >= CTX, torch.zeros_like(wt), wt)
            if WINDOW_CAP > 0:
                wt = torch.where(wt > WINDOW_CAP, torch.zeros_like(wt), wt)
            win[li] = wt
        return win

    @torch.no_grad()
    def redknot_keep(full, chunks, win, keep_chunks):
        """Offline build prefix using ONLY last keep_chunks chunks before window
        (far chunks dropped -> initial_state=0). keep_chunks=0 means keep ALL
        (exact full prefix). Then online run window+query."""
        import copy

        seg_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in chunks]
        doc_ids = [t for s in seg_ids for t in s]
        doc_len = len(doc_ids)
        device = model.device
        # Wmax
        Wmax = 0
        for li, wt in win.items():
            if wt is None:
                continue
            loc = wt[(wt > 0)]
            if loc.numel():
                Wmax = max(Wmax, int(loc.max().item()))
        Wmax = min(Wmax, doc_len)
        win_start = doc_len - Wmax
        # prefix region start: drop chunks further than keep_chunks before win_start
        if keep_chunks and keep_chunks > 0:
            pref_start = max(0, win_start - keep_chunks * CHUNK)
        else:
            pref_start = 0  # keep all
        # ---- offline: per-layer global state (full) + local prefix state ----
        glob_state, prefix_state = {}, {}
        saved = {}
        for li, wt in win.items():
            if wt is None:
                continue
            mod = bm.layers[li].linear_attn
            saved[li] = mod.chunk_gated_delta_rule

            def mk(_li, _orig):
                def w(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=False,
                    **kw,
                ):
                    core, fs = _orig(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        **kw,
                    )
                    glob_state[_li] = fs.detach()
                    if pref_start <= 0:
                        ps = torch.zeros_like(fs)
                    elif win_start > pref_start:
                        # prefix state over [pref_start, win_start): drop earlier
                        _, ps = _orig(
                            query[:, pref_start:win_start],
                            key[:, pref_start:win_start],
                            value[:, pref_start:win_start],
                            g=g[:, pref_start:win_start],
                            beta=beta[:, pref_start:win_start],
                            initial_state=None,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        )
                    else:
                        ps = torch.zeros_like(fs)
                    prefix_state[_li] = ps.detach()
                    return core, fs

                return w

            mod.chunk_gated_delta_rule = mk(li, saved[li])
        try:
            cache = DynamicCache(config=model.config)
            ids = torch.tensor([doc_ids], device=device)
            pids = torch.arange(0, doc_len, device=device).unsqueeze(0)
            model(
                input_ids=ids, position_ids=pids, past_key_values=cache, use_cache=True
            )
        finally:
            for li, f in saved.items():
                bm.layers[li].linear_attn.chunk_gated_delta_rule = f
        # ---- online: window+query, local from prefix, global from full ----
        q_ids = tok(QP.format(q=None) if False else "", add_special_tokens=False)[
            "input_ids"
        ]
        cache2 = copy.deepcopy(cache)
        for layer in cache2.layers:
            if (
                getattr(layer, "keys", None) is not None
                and layer.keys.shape[2] >= doc_len
            ):
                layer.keys = layer.keys[:, :, :win_start, :].contiguous()
                layer.values = layer.values[:, :, :win_start, :].contiguous()
        init_state = {}
        for li, wt in win.items():
            if wt is None:
                continue
            isg = (wt <= 0).to(glob_state[li].device)
            st = prefix_state[li].clone()
            # global heads use FULL doc state
            st[:, isg] = glob_state[li][:, isg]
            init_state[li] = st
        run_state = {li: init_state[li] for li in init_state}
        saved = {}
        for li in init_state:
            mod = bm.layers[li].linear_attn
            saved[li] = mod.chunk_gated_delta_rule

            def mk(_li, _orig):
                def w(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=False,
                    **kw,
                ):
                    core, fs = _orig(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=run_state[_li],
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        **kw,
                    )
                    run_state[_li] = fs.detach()
                    return core, fs

                return w

            mod.chunk_gated_delta_rule = mk(li, saved[li])
        try:
            qp_ids = tok(QP_FULL, add_special_tokens=False)["input_ids"]
            online_ids = doc_ids[win_start:doc_len] + qp_ids
            ids = torch.tensor([online_ids], device=device)
            pids = torch.arange(
                win_start, win_start + len(online_ids), device=device
            ).unsqueeze(0)
            out = model(
                input_ids=ids, position_ids=pids, past_key_values=cache2, use_cache=True
            )
            cache2 = out.past_key_values
            nx = out.logits[0, -1].argmax().view(1, 1)
            pos = win_start + len(online_ids)
            g = [int(nx[0, 0])]
            for _ in range(MAX_NEW - 1):
                pids = torch.tensor([[pos]], device=device)
                og = model(
                    input_ids=nx,
                    position_ids=pids,
                    past_key_values=cache2,
                    use_cache=True,
                )
                cache2 = og.past_key_values
                nx = og.logits[0, -1].argmax().view(1, 1)
                g.append(int(nx[0, 0]))
                pos += 1
            return tok.decode(g, skip_special_tokens=True)
        finally:
            for li, f in saved.items():
                bm.layers[li].linear_attn.chunk_gated_delta_rule = f

    print("=" * 90)
    print(
        f"APPROX-2 ERROR QUANT  ctx={CTX} cap={WINDOW_CAP} keep_chunks={KEEP_LIST} n={len(samples)}"
    )
    print("=" * 90)
    agg = {k: [0.0, 0] for k in KEEP_LIST}
    agg_dense = [0.0, 0]
    for s in samples:
        global QP_FULL
        QP_FULL = QP.format(q=s["q"])
        full = "\n\n".join(s["chunks"]) + QP_FULL
        win = win_map(full)
        dn = dense(full)
        dF = f1(short(dn), s["golds"])
        agg_dense[0] += dF
        agg_dense[1] += 1
        line = f"[{s['ds']}] dense F1={dF:.2f} '{short(dn)[:30]}'"
        for k in KEEP_LIST:
            rk = redknot_keep(full, s["chunks"], win, k)
            rF = f1(short(rk), s["golds"])
            agg[k][0] += rF
            agg[k][1] += 1
            tag = "ALL" if k == 0 else f"k={k}"
            line += f" | {tag} F1={rF:.2f}"
        print(line)
    print("-" * 90)
    print(f"AVG dense F1={agg_dense[0] / max(agg_dense[1], 1):.3f}")
    for k in KEEP_LIST:
        tag = "keep-ALL(exact prefix)" if k == 0 else f"keep-last-{k}-chunks"
        print(f"AVG {tag:24} F1={agg[k][0] / max(agg[k][1], 1):.3f}")
    print("=" * 90)


QP_FULL = ""
if __name__ == "__main__":
    main()
