#!/usr/bin/env python3
"""Measure per-head attention sparsity and per-layer union coverage (memory-safe).

For every (layer, head) on a real long-context prompt:
  1. take the attention distribution of the last NQ query positions over all keys,
  2. find the smallest set of key tokens whose cumulative mass reaches THRESH
     (default 0.99) -> that head's "essential" tokens.
Per layer, take the UNION across heads:  union_ratio[layer] = |union| / T.

Paper claim: shallow layers' heads attend to *different* tokens -> large union;
deep layers concentrate -> union ratio drops with depth.

MEMORY-SAFE & CORRECT: we monkey-patch the model's `eager_attention_forward`
(which already applies RoPE / q_norm / GQA correctly and returns attn_weights).
On each call we consume the [B,H,T,T] weights for the last NQ rows, record the
per-layer union, and let the tensor be freed right after -> peak = one layer.

Usage:
  REDKNOT_MODEL=qwen   python test/srt/redknot/measure_head_sparsity.py
  REDKNOT_MODEL=llama  python test/srt/redknot/measure_head_sparsity.py
Env: REDKNOT_CTX(8192) REDKNOT_N_SAMPLES(2) REDKNOT_THRESH(0.99) REDKNOT_NQ(8)
     REDKNOT_DTYPE(int4|bf16) REDKNOT_DATASETS(hotpotqa)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODELS = {
    "qwen": "/mnt/tidal-alsh01/dataset/redone/096/models/Qwen3-32B",
    "llama": "/mnt/tidal-alsh01/dataset/redone/096/models/Llama-3.3-70B-Instruct",
}
LONGBENCH = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/LongBench/data",
)
MODEL_KEY = os.environ.get("REDKNOT_MODEL", "qwen")
MODEL_PATH = os.environ.get("REDKNOT_MODEL_PATH", MODELS[MODEL_KEY])
CTX = int(os.environ.get("REDKNOT_CTX", "8192"))
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "2"))
THRESH = float(os.environ.get("REDKNOT_THRESH", "0.99"))
NQ = int(os.environ.get("REDKNOT_NQ", "8"))
DTYPE_MODE = os.environ.get("REDKNOT_DTYPE", "int4")

# global accumulator the patched attention writes into
_LAYER_RESULTS = []  # list of (union_ratio, headmean_ratio) in call order


def _consume_attn_weights(attn_weights, thresh, nq):
    """attn_weights: [B, H, T, T] (softmaxed). Record, for the last nq query rows:
    - union_ratio:    |union over heads of per-head essential tokens| / T
    - headmean_ratio: mean per-head essential-token ratio
    - massfrac_ratio: layer-level token-mass sparsity. Aggregate attention over
      ALL heads (and the nq query rows) into a single per-token importance
      score, then the smallest set of tokens whose cumulative aggregated mass
      reaches `thresh`, as a fraction of T. This is the Sparse-FFN selector:
      the fraction of tokens that carry 90% of the layer's total attention.
    """
    a = attn_weights[0]  # [H, T, T]
    H, T, _ = a.shape
    qs = max(0, T - nq)
    sub = a[:, qs:, :].float()  # [H, nq, T]
    thr = torch.tensor(thresh, device=a.device)

    # ---- per-head union (head heterogeneity) ----
    union = torch.zeros(T, dtype=torch.bool, device=a.device)
    head_counts = []
    for h in range(H):
        ess = torch.zeros(T, dtype=torch.bool, device=a.device)
        for qi in range(sub.shape[1]):
            row = sub[h, qi]
            sv, si = torch.sort(row, descending=True)
            csum = torch.cumsum(sv, dim=0)
            k = int(torch.searchsorted(csum, thr).item()) + 1
            k = min(k, T)
            ess[si[:k]] = True
        head_counts.append(int(ess.sum().item()))
        union |= ess
    headmean = sum(c / T for c in head_counts) / H
    headmax = max(head_counts) / T  # the greediest head in this layer (Paged block)
    # p75: a robust "Paged block size" — the layer's 75th-percentile head reach,
    # less extreme than max, used as the realistic Paged per-layer reservation.
    _sorted = sorted(head_counts)
    _p75_idx = min(len(_sorted) - 1, int(round(0.75 * (len(_sorted) - 1))))
    headp75 = _sorted[_p75_idx] / T

    # ---- layer-level token-mass sparsity (Sparse-FFN selector) ----
    # aggregate attention mass each key token receives, over all heads & query rows
    mass = sub.sum(dim=(0, 1))  # [T]
    mass = mass / mass.sum().clamp_min(1e-9)
    sv, _ = torch.sort(mass, descending=True)
    csum = torch.cumsum(sv, dim=0)
    km = int(torch.searchsorted(csum, thr).item()) + 1
    km = min(km, T)
    massfrac = km / T

    _LAYER_RESULTS.append(
        (int(union.sum().item()) / T, headmean, massfrac, headmax, headp75)
    )


def load_prompts(tok, ds_name, n_samples, ctx_tokens):
    path = os.path.join(LONGBENCH, f"{ds_name}.jsonl")
    raw = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("context") and r.get("input"):
                raw.append(r)
    prompts = []
    for i in range(len(raw)):
        if len(prompts) >= n_samples:
            break
        ids = tok(raw[i]["context"], add_special_tokens=False)["input_ids"]
        j = i + 1
        while len(ids) < ctx_tokens and j < len(raw):
            ids = ids + tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
            j += 1
        ids = ids[:ctx_tokens]
        qids = tok(
            "\n\nQuestion: " + raw[i]["input"] + "\nAnswer:", add_special_tokens=False
        )["input_ids"]
        prompts.append(ids + qids)
    return prompts


def patch_attention(model_key, thresh, nq):
    """Monkey-patch the model family's eager_attention_forward to record stats."""
    if model_key == "qwen":
        import transformers.models.qwen3.modeling_qwen3 as M
    else:
        import transformers.models.llama.modeling_llama as M
    orig = M.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, attn_weights = orig(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kw
        )
        try:
            _consume_attn_weights(attn_weights.detach(), thresh, nq)
        except Exception as e:  # noqa: BLE001
            print("  [warn] consume failed:", e)
        return out, attn_weights

    M.eager_attention_forward = patched
    return M, orig


@torch.no_grad()
def measure(model, input_ids):
    global _LAYER_RESULTS
    _LAYER_RESULTS = []
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    T = ids.shape[1]
    model(input_ids=ids, use_cache=False)
    L = len(_LAYER_RESULTS)
    union = [_LAYER_RESULTS[i][0] for i in range(L)]
    headmean = [_LAYER_RESULTS[i][1] for i in range(L)]
    massfrac = [_LAYER_RESULTS[i][2] for i in range(L)]
    headmax = [_LAYER_RESULTS[i][3] for i in range(L)]
    headp75 = [_LAYER_RESULTS[i][4] for i in range(L)]
    return union, headmean, massfrac, headmax, headp75, T, L


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(
        f"[sparsity] model={MODEL_KEY} ctx={CTX} n={N_SAMPLES} thr={THRESH} nq={NQ} dtype={DTYPE_MODE}"
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    patch_attention(MODEL_KEY, THRESH, NQ)  # patch BEFORE load so model picks eager fn

    if DTYPE_MODE == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()
    else:
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=qc,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()

    datasets = os.environ.get("REDKNOT_DATASETS", "hotpotqa").split(",")
    prompts = []
    for ds in datasets:
        prompts += load_prompts(tok, ds.strip(), N_SAMPLES, CTX)
    print(f"[sparsity] {len(prompts)} prompts")

    all_u, all_h, all_m, all_x, all_p, Lref = [], [], [], [], [], None
    for pi, p in enumerate(prompts):
        u, h, m, xmax, xp75, T, L = measure(model, p)
        Lref = L
        all_u.append(u)
        all_h.append(h)
        all_m.append(m)
        all_x.append(xmax)
        all_p.append(xp75)
        print(
            f"[sparsity] sample {pi}: T={T} L={L} "
            f"union[0]={u[0]:.3f} union[-1]={u[-1]:.3f} | "
            f"massfrac[0]={m[0]:.3f} mid={m[L // 2]:.3f} [-1]={m[-1]:.3f}"
        )
        torch.cuda.empty_cache()

    import statistics

    L = Lref
    mu = [statistics.mean(s[li] for s in all_u) for li in range(L)]
    mh = [statistics.mean(s[li] for s in all_h) for li in range(L)]
    mm = [statistics.mean(s[li] for s in all_m) for li in range(L)]
    mx = [statistics.mean(s[li] for s in all_x) for li in range(L)]
    mp = [statistics.mean(s[li] for s in all_p) for li in range(L)]
    result = {
        "model": MODEL_KEY,
        "model_path": MODEL_PATH,
        "ctx": CTX,
        "threshold": THRESH,
        "nq": NQ,
        "n_prompts": len(prompts),
        "num_layers": L,
        "layer_union_ratio": mu,
        "layer_headmean_ratio": mh,
        "layer_headmax_ratio": mx,
        "layer_headp75_ratio": mp,
        "layer_massfrac_ratio": mm,
        "per_sample_union": all_u,
        "per_sample_massfrac": all_m,
    }
    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / f"head_sparsity_{MODEL_KEY}.json"
    json.dump(result, open(outpath, "w"), indent=2)
    print(f"\n[sparsity] wrote {outpath}")
    print("\n layer  union  headmean  massfrac")
    for li in range(L):
        print(f"  {li:3d}   {mu[li]:.4f}  {mh[li]:.4f}   {mm[li]:.4f}")


if __name__ == "__main__":
    main()
