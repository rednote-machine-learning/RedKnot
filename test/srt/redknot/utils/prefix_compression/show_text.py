#!/usr/bin/env python3
"""Print the ACTUAL generated text: baseline vs trimmed configs side by side.

Answers the question: is the decoded text the same after KV trimming?
Runs one prefill, then greedy-decodes from:
  - baseline       (full KV)
  - trim<32,sink128 (near-lossless config)
  - trim<64,sink4   (pure per-head, profile's intent)
"""

import os, sys, gc
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_swa_pd_2gpu as B

MODEL_PATH = B.MODEL_PATH
PROFILE = os.environ.get(
    "SWA_PROFILE_PATH",
    "/mnt/tidal-alsh01/dataset/redone/RedKnotV0.2/test/srt/redknot/head_class/qwen3-32B_optimal_g15_lf_ret.json",
)
PLEN = int(os.environ.get("SWA_PLEN", "8192"))
NEW = int(os.environ.get("SWA_NEW", "64"))
DEV = "cuda:0"


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    local_mask, _ = B.load_local_head_mask(PROFILE)
    print(f"loading model bf16 2GPU ... prefix={PLEN}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "40GiB", 1: "40GiB"},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    ids = B._make_prefix_ids(tok, PLEN).to(DEV)
    q = tok(
        "\n\nBased on the above text, briefly summarize the key points:\n",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(DEV)
    inp = torch.cat([ids, q], 1)
    total = inp.shape[1]

    out = model(input_ids=inp, use_cache=True)
    first = int(out.logits[0, -1, :].argmax())

    configs = [
        ("baseline (full KV)", None, None),
        ("trim<32, sink=128 (near-lossless)", 32, 128),
        ("trim<64, sink=4  (pure per-head)", 64, 4),
    ]
    results = {}
    for name, tlm, sink in configs:
        cache = B._clone_cache(out.past_key_values)
        if tlm is not None:
            B.TRIM_LAYER_MAX = tlm
            B.trim_and_transfer(cache, total, local_mask, 4096, sink, B.DECODE_DEV)
        toks, _ = B.greedy_decode(model, cache, first, tok, NEW, DEV)
        txt = tok.decode(toks, skip_special_tokens=True)
        results[name] = txt
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    base = results["baseline (full KV)"]
    print("\n" + "=" * 100)
    for name, txt in results.items():
        same = "  [IDENTICAL to baseline]" if txt == base else "  [DIFFERS]"
        if name.startswith("baseline"):
            same = ""
        print(f"\n### {name}{same}")
        print(txt)
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
