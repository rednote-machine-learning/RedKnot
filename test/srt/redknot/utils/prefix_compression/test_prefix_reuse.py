#!/usr/bin/env python3
"""Prefix-KV reuse + concatenated text: offline-trim, online-reuse pipeline.

Real deployment flow this models:
  OFFLINE:  full-prefill the prefix -> per-head trim local heads to sink+window
            -> store the trimmed prefix KV.
  ONLINE:   load (reuse) the trimmed prefix KV; prefill an EQUAL-LENGTH new text
            chunk on top of it (the new text attends to the reused trimmed prefix
            KV); then decode.

Accuracy is compared against the no-reuse baseline: [prefix + new_text] fully
prefilled together with NO trimming, then decoded.

True eviction is realized with a per-(layer,kv-head) -inf mask before softmax
(equivalent to "the trimmed KV is not stored"), applied to BOTH:
  * the new-text prefill chunk (its queries cannot see the trimmed prefix middle
    for local heads), and
  * the decode steps.

Position handling for reuse: the new text's positions continue from prefix_len
(RoPE/cache_position), even though local heads physically keep fewer entries.

Config: trim<32, sink=128, window=4096, dense Qwen3-32B.
new_text length == prefix length (equal-length).

Usage:
  CUDA_VISIBLE_DEVICES=0,1 SWA_PREFIX_LENGTHS=4096,8192,12288 python test_prefix_reuse.py
"""

from __future__ import annotations

import gc, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE = os.environ.get(
    "SWA_PROFILE_PATH",
    "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
)
PREFIX_LENGTHS = [
    int(x) for x in os.environ.get("SWA_PREFIX_LENGTHS", "4096,8192,12288").split(",")
]
WINDOW, SINK, TRIM_LAYER_MAX = 4096, 128, 32
NEW = int(os.environ.get("SWA_NEW", "48"))
DEV = "cuda:0"

# eviction state read by the custom attention
_EV = {
    "on": False,
    "local_mask": None,
    "prefix_len": 0,
    "window": WINDOW,
    "sink": SINK,
    "trim_layer_max": TRIM_LAYER_MAX,
    "nkv": 8,
}


def cos(a, b):
    return F.cosine_similarity(
        a.double().flatten().unsqueeze(0), b.double().flatten().unsqueeze(0)
    ).item()


def make_attn(layer_idx_attr="layer_idx"):
    """SDPA attention that evicts trimmed prefix KV for local heads via -inf.

    Eviction region for a local head = prefix positions [sink, prefix_len-window)
    i.e. everything in the *prefix* outside sink+window.  Applies whenever the
    key length covers the prefix (both the new-text prefill chunk and decode).
    """

    def attn_fwd(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        k = repeat_kv(key, module.num_key_value_groups)
        v = repeat_kv(value, module.num_key_value_groups)
        Bsz, Hq, qlen, _ = query.shape
        kvlen = k.shape[-2]

        li = getattr(module, layer_idx_attr, None)
        do_evict = False
        if _EV["on"] and li is not None and li < _EV["trim_layer_max"]:
            P = _EV["prefix_len"]
            w_start = max(_EV["sink"], P - _EV["window"])
            if w_start > _EV["sink"] and kvlen >= P:
                do_evict = any(
                    bool(_EV["local_mask"][li][h]) for h in range(_EV["nkv"])
                )

        # Base causal mask as a compact [1,1,q,kv] (broadcast over heads). We
        # need an explicit mask because reuse has q_len != kv_len (SDPA
        # is_causal misaligns). Query local index i -> absolute pos kvlen-qlen+i.
        if attention_mask is not None:
            base = attention_mask[:, :, :, :kvlen]  # [B,1or H,q,kv]
        else:
            q_abs = torch.arange(kvlen - qlen, kvlen, device=query.device).unsqueeze(1)
            k_abs = torch.arange(kvlen, device=query.device).unsqueeze(0)
            base = torch.zeros(qlen, kvlen, dtype=query.dtype, device=query.device)
            base = base.masked_fill(k_abs > q_abs, float("-inf"))
            base = base.unsqueeze(0).unsqueeze(0)  # [1,1,q,kv]

        if not do_evict:
            # cheap path: broadcast mask, no per-head expansion
            out = F.scaled_dot_product_attention(
                query, k, v, attn_mask=base, scale=scaling, dropout_p=0.0
            )
            return out.transpose(1, 2).contiguous(), None

        # eviction path: per-head mask only for this (trimmable) layer
        nkv = _EV["nkv"]
        grp = Hq // nkv
        lm = _EV["local_mask"][li]
        add = base.expand(Bsz, Hq, qlen, kvlen).clone()
        for kvh in range(nkv):
            if bool(lm[kvh]):
                a, b2 = kvh * grp, (kvh + 1) * grp
                add[:, a:b2, :, _EV["sink"] : w_start] = float("-inf")
        out = F.scaled_dot_product_attention(
            query, k, v, attn_mask=add, scale=scaling, dropout_p=0.0
        )
        return out.transpose(1, 2).contiguous(), None

    return attn_fwd


def build_prefix_and_text(tok, plen):
    """Return prefix_ids (len=plen) and new_text_ids (len≈plen), distinct content."""
    prefix_seed = (
        "The history of artificial intelligence began in antiquity with "
        "myths of artificial beings. Philosophers described thought as "
        "symbolic manipulation, leading to the digital computer. "
    )
    text_seed = (
        "Quantum computing exploits superposition and entanglement to "
        "process information. Qubits differ fundamentally from classical "
        "bits, enabling algorithms like Shor's and Grover's. "
    )
    pid = tok(prefix_seed, add_special_tokens=False)["input_ids"]
    tid = tok(text_seed, add_special_tokens=False)["input_ids"]
    pre = (pid * (plen // len(pid) + 2))[:plen]
    txt = (tid * (plen // len(tid) + 2))[:plen]
    return torch.tensor([pre]), torch.tensor([txt])


@torch.no_grad()
def greedy(model, cache, first, tok, n, dev, prefix_len):
    """Decode keeping eviction on; position continues via cache_position."""
    nxt = torch.tensor([[first]], device=dev)
    toks = [first]
    logs = []
    seqlen = cache.get_seq_length()
    for _ in range(n - 1):
        cp = torch.tensor([seqlen], device=dev)
        o = model(
            input_ids=nxt, past_key_values=cache, use_cache=True, cache_position=cp
        )
        cache = o.past_key_values
        seqlen += 1
        lg = o.logits[0, -1, :]
        logs.append(lg.detach().cpu().clone())
        nxt = lg.argmax().view(1, 1)
        t = int(nxt[0, 0])
        toks.append(t)
        if t == tok.eos_token_id:
            break
    return toks, logs


@torch.no_grad()
def run(model, tok, plen, local_mask):
    pre_ids, txt_ids = build_prefix_and_text(tok, plen)
    pre_ids, txt_ids = pre_ids.to(DEV), txt_ids.to(DEV)
    P, T = pre_ids.shape[1], txt_ids.shape[1]
    full_ids = torch.cat([pre_ids, txt_ids], 1)

    # ---------- BASELINE: full prefill [prefix+text], NO trim ----------
    _EV["on"] = False
    ob = model(input_ids=full_ids, use_cache=True)
    first_b = int(ob.logits[0, -1, :].argmax())
    base_tok, base_log = greedy(model, ob.past_key_values, first_b, tok, NEW, DEV, P)
    base_txt = tok.decode(base_tok, skip_special_tokens=True)
    del ob
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- REUSE: offline trim prefix, online reuse + text prefill ----------
    _EV.update({"on": True, "local_mask": local_mask, "prefix_len": P})
    # 1) offline: prefill prefix only (full), keep its cache (eviction handled by mask online)
    o1 = model(
        input_ids=pre_ids, use_cache=True
    )  # prefix KV (full tensor; evict via mask)
    cache = o1.past_key_values
    # 2) online: prefill the new text on top, positions continue from P
    cp = torch.arange(P, P + T, device=DEV)
    o2 = model(
        input_ids=txt_ids, past_key_values=cache, use_cache=True, cache_position=cp
    )
    cache = o2.past_key_values
    first_r = int(o2.logits[0, -1, :].argmax())
    # 3) decode
    reuse_tok, reuse_log = greedy(model, cache, first_r, tok, NEW, DEV, P)
    reuse_txt = tok.decode(reuse_tok, skip_special_tokens=True)
    _EV["on"] = False
    del o1, o2, cache
    gc.collect()
    torch.cuda.empty_cache()

    n = min(len(base_log), len(reuse_log))
    sc = [cos(base_log[i], reuse_log[i]) for i in range(n)]
    m = min(len(base_tok), len(reuse_tok))
    match = sum(1 for i in range(m) if base_tok[i] == reuse_tok[i]) / m if m else 1.0
    return {
        "prefix_len": P,
        "text_len": T,
        "total": P + T,
        "cos_d1": sc[0] if sc else 1.0,
        "avg_cos": sum(sc) / len(sc) if sc else 1.0,
        "token_match": match,
        "base_txt": base_txt[:120],
        "reuse_txt": reuse_txt[:120],
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    print("=" * 92)
    print(" PREFIX-KV REUSE + equal-length text (trim<32, sink=128, dense Qwen3-32B)")
    print(f" prefixes={PREFIX_LENGTHS} (new text == prefix length)")
    print("=" * 92)

    local_mask, _ = B.load_local_head_mask(PROFILE)
    _EV["nkv"] = local_mask.shape[1]
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ALL_ATTENTION_FUNCTIONS["reuse_evict"] = make_attn()

    print(" loading model bf16 2GPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="reuse_evict",
    ).eval()
    print(" model loaded.\n")

    rows = []
    for plen in PREFIX_LENGTHS:
        print(f"[prefix={plen} + text={plen} = {2 * plen}]")
        try:
            r = run(model, tok, plen, local_mask)
        except torch.cuda.OutOfMemoryError:
            print("  OOM")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(
            f"  reuse-trim vs full-baseline: cos_d1={r['cos_d1']:.4f} "
            f"avg={r['avg_cos']:.4f} token_match={r['token_match']:.1%}"
        )
        print(f"   base : {r['base_txt'][:80]}")
        print(f"   reuse: {r['reuse_txt'][:80]}")

    print(
        f"\n{'=' * 92}\n SUMMARY: reuse trimmed prefix + equal-length text vs full no-reuse\n{'=' * 92}"
    )
    print(
        f"  {'prefix':>7} {'text':>7} {'total':>7} {'cos_d1':>9} {'avg_cos':>9} {'tok_match':>9}"
    )
    for r in rows:
        print(
            f"  {r['prefix_len']:>7} {r['text_len']:>7} {r['total']:>7} "
            f"{r['cos_d1']:>9.4f} {r['avg_cos']:>9.4f} {r['token_match']:>8.1%}"
        )

    out = Path(__file__).with_suffix(".results.json")
    json.dump(
        {
            "config": {
                "trim_layer_max": TRIM_LAYER_MAX,
                "sink": SINK,
                "window": WINDOW,
                "new_text": "equal-length",
            },
            "rows": rows,
        },
        open(out, "w"),
        indent=2,
        default=str,
    )
    print(f"\n results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
