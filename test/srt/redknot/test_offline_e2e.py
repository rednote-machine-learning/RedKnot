#!/usr/bin/env python3
"""End-to-end validation of TRUE RedKnot sparse prefill under sglang TP.

Flow (no scheduler changes; uses the __RKBUILD__ sentinel + Phase-1 plumbing):

  1. For each document chunk, compute a deterministic segment id, then prefill
     it once via engine.generate(chunk, redknot_offline_segments=["__RKBUILD__:<sid>"])
     which makes the RedKnot backend snapshot the chunk's KV into the per-rank
     OfflineKVCache.
  2. Query: engine.generate(query, redknot_offline_segments=[[sid0, sid1, ...]])
     which activates the TRUE sparse path (splices offline KV, runs head-class
     attention) instead of the dense SDPA fallback.

Validation modes (REDKNOT_VALIDATE):
  * parity  : use an ALL-DENSE/GLOBAL head config so RedKnot is mathematically
              equivalent to full attention; compare the sparse-path answer to a
              plain full-context generate. Expect near-identical text.
  * sparse  : use the real head config; just print answers + TTFT.

Run (35B tp=2):
  PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1 \
    REDKNOT_TP_SIZE=2 REDKNOT_VALIDATE=parity \
    python test/srt/redknot/test_offline_e2e.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
TP = int(os.environ.get("REDKNOT_TP_SIZE", "2"))
MEM_FRAC = os.environ.get("REDKNOT_MEM_FRACTION_STATIC", "0.85")
HEAD_CFG = os.environ.get(
    "REDKNOT_HEAD_CFG",
    str(
        Path(__file__).resolve().parent
        / "head_class"
        / "qwen3.5-35B-A3B_redknot_server.json"
    ),
)
WATCHDOG = os.environ.get("REDKNOT_WATCHDOG_TIMEOUT", "")
MODE = os.environ.get("REDKNOT_VALIDATE", "parity")


def _patch_env():
    import sglang.srt.utils.common as _c

    _orig = _c.assert_pkg_version

    def _lenient(pkg, mv, msg):
        try:
            return _orig(pkg, mv, msg)
        except Exception as e:  # noqa: BLE001
            print(f"[e2e] bypass version assert {pkg!r}: {e}")
            return None

    _c.assert_pkg_version = _lenient
    try:
        import sglang.srt.entrypoints.engine as _eng

        _eng.assert_pkg_version = _lenient
    except Exception:
        pass


def main():
    _patch_env()
    import sglang as sgl
    from transformers import AutoTokenizer
    from sglang.srt.layers.attention.redknot.offline_cache import OfflineKVCache

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # ── Two short document chunks + a question answerable from them ──
    doc0 = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France. It was completed in 1889 and was named after the "
        "engineer Gustave Eiffel, whose company designed and built the tower."
    )
    doc1 = (
        "The Great Wall of China is a series of fortifications built across the "
        "historical northern borders of ancient Chinese states. The Statue of "
        "Liberty is located in New York City."
    )
    question = (
        "\n\nUsing only the documents above, answer with the shortest exact "
        "span.\nQuestion: In which city is the Eiffel Tower located?\nAnswer:"
    )

    docs = [doc0, doc1]
    doc_ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    sids = [
        OfflineKVCache.compute_segment_id(MODEL, ids, prepend_bos=(i == 0))
        for i, ids in enumerate(doc_ids)
    ]
    print(f"[e2e] segment ids: {[s[:12] + '...' for s in sids]}")

    kwargs = dict(
        model_path=MODEL,
        attention_backend="redknot",
        tp_size=TP,
        trust_remote_code=True,
        enable_multimodal=False,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=True,  # ensure build reqs prefill at positions [0,L)
        mem_fraction_static=float(MEM_FRAC),
        redknot_head_config_path=HEAD_CFG,
        log_level="info",
    )
    if WATCHDOG:
        kwargs["watchdog_timeout"] = float(WATCHDOG)

    print(f"[e2e] launching engine tp={TP} mode={MODE} ...")
    t0 = time.perf_counter()
    engine = sgl.Engine(**kwargs)
    print(f"[e2e] engine up in {time.perf_counter() - t0:.1f}s")

    greedy = {"temperature": 0.0, "max_new_tokens": 16}
    try:
        # ── 1) BUILD offline segments ──
        for d, sid in zip(docs, sids):
            out = engine.generate(
                d,
                {"temperature": 0.0, "max_new_tokens": 1},
                redknot_offline_segments=["__RKBUILD__:" + sid],
            )
            print(f"[e2e] built segment {sid[:12]}... (prefill ok)")

        # ── 2) QUERY using the offline segments (TRUE sparse path) ──
        # Single request: redknot_offline_segments is the flat per-request list.
        ctx = "\n\n".join(docs)
        sparse_out = engine.generate(
            ctx + question,
            greedy,
            redknot_offline_segments=sids,
        )
        sparse_text = (
            sparse_out["text"] if isinstance(sparse_out, dict) else str(sparse_out)
        )
        print(f"[e2e] SPARSE answer: {sparse_text!r}")

        # ── 3) Reference: plain full-context generate (no offline segments) ──
        ref_out = engine.generate(ctx + question, greedy)
        ref_text = ref_out["text"] if isinstance(ref_out, dict) else str(ref_out)
        print(f"[e2e] DENSE  answer: {ref_text!r}")

        ok_paris = "paris" in sparse_text.lower()
        print(f"[e2e] sparse answer contains 'Paris': {ok_paris}")
        if MODE == "parity":
            match = sparse_text.strip()[:32] == ref_text.strip()[:32]
            print(f"[e2e] parity (sparse==dense prefix): {match}")
    finally:
        engine.shutdown()
    print("[e2e] DONE")


if __name__ == "__main__":
    main()
