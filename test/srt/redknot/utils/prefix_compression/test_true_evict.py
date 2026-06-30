#!/usr/bin/env python3
"""True KV eviction (not zero-fill) for per-head prefix compression.

The online design is: after offline full prefill, the serving node does NOT
store KV outside [sink] U [window] for local heads -- those entries simply do
not exist and never enter attention.

Earlier experiments approximated this by ZERO-FILLING the trimmed positions
(they stay in the dense tensor with value 0). That is not the same as eviction:
a zeroed key K=0 still scores q.K=0 and receives exp(0)=1 weight in the softmax
denominator, diluting the kept positions. TRUE eviction removes them from the
softmax entirely.

We realize true-eviction *semantics* exactly by masking the trimmed positions
with -inf before softmax, per (layer, kv-head), via a custom attention function.
This is mathematically identical to the KV not being stored (it contributes
nothing to softmax). Memory savings are computed separately (those positions
would not occupy memory in a real engine).

Three-way comparison vs full-KV baseline:
  * zero-fill   : trimmed positions set to 0 (old approximation)
  * true-evict  : trimmed positions masked -inf (real "not stored" semantics)

Config: trim<32, sink=128, window=4096, dense Qwen3-32B.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 SWA_PREFIX_LENGTHS=8192,16384 python test_true_evict.py
"""

from __future__ import annotations

import gc, json, math, os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE = os.environ.get(
    "SWA_PROFILE_PATH",
    "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
)
PREFIX_LENGTHS = [
    int(x) for x in os.environ.get("SWA_PREFIX_LENGTHS", "8192,16384,32768").split(",")
]
WINDOW, SINK, TRIM_LAYER_MAX = 4096, 128, 32
NEW = int(os.environ.get("SWA_NEW", "48"))
DEV = "cuda:0"

# ---- global state the custom attention reads ----
_EVICT = {
    "on": False,
    "local_mask": None,
    "seq_len": 0,
    "window": WINDOW,
    "sink": SINK,
    "trim_layer_max": TRIM_LAYER_MAX,
    "num_kv_heads": 8,
}


def cos(a, b):
    return F.cosine_similarity(
        a.double().flatten().unsqueeze(0), b.double().flatten().unsqueeze(0)
    ).item()


def make_evict_attention(orig_layer_idx_attr="layer_idx"):
    """SDPA-based attention that adds a per-(layer,kv-head) -inf eviction mask.

    Memory-efficient (fused SDPA kernel); the eviction is expressed as an
    additive attn_mask so trimmed positions get no softmax mass == true
    eviction. Only active during decode (q_len==1) for the trimmable layers.
    """

    def evict_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        B_, Hq, q_len, _ = query.shape
        kvlen = key_states.shape[-2]

        # base additive mask (causal etc.) sliced to kv length
        add_mask = None
        if attention_mask is not None:
            add_mask = attention_mask[:, :, :, :kvlen]

        if _EVICT["on"]:
            li = getattr(module, orig_layer_idx_attr, None)
            if li is not None and li < _EVICT["trim_layer_max"]:
                lm = _EVICT["local_mask"][li]
                S = _EVICT["seq_len"]
                w_start = max(_EVICT["sink"], S - _EVICT["window"])
                if w_start > _EVICT["sink"] and kvlen >= S:
                    nkv = _EVICT["num_kv_heads"]
                    grp = Hq // nkv
                    if add_mask is None:
                        add_mask = torch.zeros(
                            B_, Hq, q_len, kvlen, dtype=query.dtype, device=query.device
                        )
                    else:
                        add_mask = add_mask.expand(B_, Hq, q_len, kvlen).clone()
                    for kvh in range(nkv):
                        if bool(lm[kvh]):
                            qh0, qh1 = kvh * grp, (kvh + 1) * grp
                            add_mask[:, qh0:qh1, :, _EVICT["sink"] : w_start] = float(
                                "-inf"
                            )

        is_causal = add_mask is None and q_len > 1
        out = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=add_mask,
            scale=scaling,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous()
        return out, None

    return evict_attention_forward


@torch.no_grad()
def greedy(model, cache, first, tok, n, dev):
    nxt = torch.tensor([[first]], device=dev)
    toks = [first]
    logs = []
    for _ in range(n - 1):
        o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
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
    ids = B._make_prefix_ids(tok, plen).to(DEV)
    q = tok(
        "\n\nBased on the above text, briefly summarize the key points:\n",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(DEV)
    inp = torch.cat([ids, q], 1)
    total = inp.shape[1]

    # baseline (full attention, no evict)
    _EVICT["on"] = False
    o = model(input_ids=inp, use_cache=True)
    first = int(o.logits[0, -1, :].argmax())
    base_cache = B._clone_cache(o.past_key_values)
    base_tok, base_log = greedy(model, base_cache, first, tok, NEW, DEV)
    base_txt = tok.decode(base_tok, skip_special_tokens=True)
    del base_cache
    gc.collect()
    torch.cuda.empty_cache()

    # zero-fill (old approximation): physically zero, no evict mask
    _EVICT["on"] = False
    zf_cache = B._clone_cache(o.past_key_values)
    B.TRIM_LAYER_MAX = TRIM_LAYER_MAX
    B.trim_and_transfer(zf_cache, total, local_mask, WINDOW, SINK, B.DECODE_DEV)
    zf_tok, zf_log = greedy(model, zf_cache, first, tok, NEW, DEV)
    zf_txt = tok.decode(zf_tok, skip_special_tokens=True)
    del zf_cache
    gc.collect()
    torch.cuda.empty_cache()

    # true-evict (mask -inf, KV kept full but excluded from softmax)
    _EVICT.update({"on": True, "local_mask": local_mask, "seq_len": total})
    te_cache = B._clone_cache(o.past_key_values)  # full KV, eviction via mask
    te_tok, te_log = greedy(model, te_cache, first, tok, NEW, DEV)
    te_txt = tok.decode(te_tok, skip_special_tokens=True)
    _EVICT["on"] = False
    del te_cache, o
    gc.collect()
    torch.cuda.empty_cache()

    def metrics(a_tok, a_log, name):
        n = min(len(base_log), len(a_log))
        c = [cos(base_log[i], a_log[i]) for i in range(n)]
        m = min(len(base_tok), len(a_tok))
        match = sum(1 for i in range(m) if base_tok[i] == a_tok[i]) / m if m else 1.0
        return {
            "cos_d1": c[0] if c else 1.0,
            "avg_cos": sum(c) / len(c) if c else 1.0,
            "token_match": match,
        }

    return {
        "prefix_len": plen,
        "total_len": total,
        "zero_fill": metrics(zf_tok, zf_log, "zf"),
        "true_evict": metrics(te_tok, te_log, "te"),
        "base_txt": base_txt[:120],
        "zf_txt": zf_txt[:120],
        "te_txt": te_txt[:120],
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as mq

    print("=" * 92)
    print(" TRUE EVICTION vs ZERO-FILL (trim<32, sink=128, dense Qwen3-32B)")
    print(f" prefixes={PREFIX_LENGTHS}")
    print("=" * 92)

    local_mask, _ = B.load_local_head_mask(PROFILE)
    _EVICT["num_kv_heads"] = local_mask.shape[1]
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # register custom attention and force eager so it is used
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["evict_eager"] = make_evict_attention()

    print(" loading model bf16 2GPU (attn=evict_eager) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="evict_eager",
    ).eval()
    print(" model loaded.\n")

    rows = []
    for plen in PREFIX_LENGTHS:
        print(f"[prefix={plen}]")
        try:
            r = run(model, tok, plen, local_mask)
        except torch.cuda.OutOfMemoryError:
            print("  OOM")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        zf, te = r["zero_fill"], r["true_evict"]
        print(
            f"  zero-fill : cos_d1={zf['cos_d1']:.4f} avg={zf['avg_cos']:.4f} match={zf['token_match']:.1%}"
        )
        print(
            f"  true-evict: cos_d1={te['cos_d1']:.4f} avg={te['avg_cos']:.4f} match={te['token_match']:.1%}"
        )

    print(
        f"\n{'=' * 92}\n SUMMARY: true-eviction vs zero-fill (vs full-KV baseline)\n{'=' * 92}"
    )
    print(
        f"  {'prefix':>7} | {'zf cos_d1':>9} {'zf match':>8} | {'te cos_d1':>9} {'te match':>8} | {'te>=zf?':>7}"
    )
    for r in rows:
        zf, te = r["zero_fill"], r["true_evict"]
        better = "YES" if te["cos_d1"] >= zf["cos_d1"] - 1e-4 else "no"
        print(
            f"  {r['prefix_len']:>7} | {zf['cos_d1']:>9.4f} {zf['token_match']:>7.1%} | "
            f"{te['cos_d1']:>9.4f} {te['token_match']:>7.1%} | {better:>7}"
        )

    out = Path(__file__).with_suffix(".results.json")
    json.dump(
        {
            "config": {
                "trim_layer_max": TRIM_LAYER_MAX,
                "sink": SINK,
                "window": WINDOW,
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
