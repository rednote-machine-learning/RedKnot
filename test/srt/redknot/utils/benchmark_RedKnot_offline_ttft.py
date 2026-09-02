#!/usr/bin/env python3
"""TRUE RedKnot sparse TTFT comparison under sglang TP (offline-segment path).

For each LongBench RAG sample:
  * BASELINE: full-context generate (dense attention), measure query TTFT.
  * REDKNOT : build each doc chunk as an offline segment (KV snapshotted into
              the per-rank OfflineKVCache), then query with those segment ids
              so the TRUE sparse path runs. Measure query TTFT (offline build
              time EXCLUDED — that is amortised across many queries in RAG).

Both share one engine (attention_backend=redknot): baseline is just a query with
no offline segments (dense SDPA), redknot is the same query WITH segments.

TTFT is measured via streaming (time to first generated token).

Run (397B-FP8 tp=8, 64K):
  PYTHONPATH=python CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    REDKNOT_MODEL_PATH=.../Qwen3.5-397B-A17B-FP8 REDKNOT_TP_SIZE=8 \
    REDKNOT_TARGET_TOKENS=64000 REDKNOT_HEAD_CFG=.../qwen3.5-397B-A17B_redknot_server.json \
    python test/srt/redknot/benchmark_RedKnot_offline_ttft.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

MODEL = os.environ.get(
    "REDKNOT_MODEL_PATH",
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/Qwen3.5-35B-A3B",
)
LONGBENCH_DIR = os.environ.get(
    "REDKNOT_LONGBENCH_DIR",
    "/mnt/tidal-alsh01/dataset/redone/xiaoyi/RedCacheV0.2/datasets/LongBench/data",
)
DATASETS = [
    x.strip()
    for x in os.environ.get("REDKNOT_DATASETS", "triviaqa").split(",")
    if x.strip()
]
N_SAMPLES = int(os.environ.get("REDKNOT_N_SAMPLES", "5"))
MAX_NEW = int(os.environ.get("REDKNOT_MAX_NEW", "16"))
TARGET_TOKENS = int(os.environ.get("REDKNOT_TARGET_TOKENS", "32000"))
CHUNK_TOKENS = int(os.environ.get("REDKNOT_CHUNK_TOKENS", "8000"))
SEED = int(os.environ.get("REDKNOT_SEED", "0"))
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


def _patch_env():
    import sglang.srt.utils.common as _c

    _orig = _c.assert_pkg_version

    def _lenient(pkg, mv, msg):
        try:
            return _orig(pkg, mv, msg)
        except Exception as e:  # noqa: BLE001
            print(f"[ttft] bypass version assert {pkg!r}: {e}")
            return None

    _c.assert_pkg_version = _lenient
    try:
        import sglang.srt.entrypoints.engine as _eng

        _eng.assert_pkg_version = _lenient
    except Exception:
        pass


def _load_samples(tok):
    out = []
    for ds in DATASETS:
        path = os.path.join(LONGBENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(path):
            continue
        raw = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("input") and r.get("context") and r.get("answers"):
                    raw.append(r)
        rng = random.Random(SEED)
        rng.shuffle(raw)
        n = len(raw)
        for i, base in enumerate(raw):
            if len([s for s in out if s["ds"] == ds]) >= N_SAMPLES:
                break
            ids = tok(base["context"], add_special_tokens=False)["input_ids"]
            j = (i + 1) % n
            while len(ids) < TARGET_TOKENS and j != i:
                ids.extend(
                    tok(raw[j]["context"], add_special_tokens=False)["input_ids"]
                )
                j = (j + 1) % n
            ids = ids[:TARGET_TOKENS]
            if len(ids) < TARGET_TOKENS:
                continue
            docs = []
            for st in range(0, len(ids), CHUNK_TOKENS):
                piece = ids[st : st + CHUNK_TOKENS]
                if len(piece) < 64:
                    break
                docs.append(tok.decode(piece, skip_special_tokens=True))
            if len(docs) < 2:
                continue
            out.append({"ds": ds, "question": base["input"], "docs": docs})
    return out


def _ttft_stream(engine, prompt, sampling, segments=None):
    t0 = time.perf_counter()
    ttft = None
    stream = engine.generate(
        prompt, sampling, stream=True, redknot_offline_segments=segments
    )
    for chunk in stream:
        piece = chunk["text"] if isinstance(chunk, dict) else str(chunk)
        if ttft is None and piece:
            ttft = time.perf_counter() - t0
    return ttft if ttft is not None else (time.perf_counter() - t0)


def main():
    _patch_env()
    import sglang as sgl
    from transformers import AutoTokenizer
    from sglang.srt.layers.attention.redknot.offline_cache import OfflineKVCache

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    samples = _load_samples(tok)
    print(
        f"[ttft] loaded {len(samples)} samples; ctx={TARGET_TOKENS} chunk={CHUNK_TOKENS}"
    )
    if not samples:
        print("[ttft] no samples; abort")
        return

    kwargs = dict(
        model_path=MODEL,
        attention_backend="redknot",
        tp_size=TP,
        trust_remote_code=True,
        enable_multimodal=False,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_radix_cache=(os.environ.get("REDKNOT_DISABLE_RADIX", "1") == "1"),
        mem_fraction_static=float(MEM_FRAC),
        redknot_head_config_path=HEAD_CFG,
        log_level=os.environ.get("REDKNOT_LOG_LEVEL", "error"),
    )
    if WATCHDOG:
        kwargs["watchdog_timeout"] = float(WATCHDOG)

    print(f"[ttft] launching engine tp={TP} ...")
    t0 = time.perf_counter()
    engine = sgl.Engine(**kwargs)
    print(f"[ttft] engine up in {time.perf_counter() - t0:.1f}s")

    q_suffix = (
        "\n\nUsing only the documents above, answer with the shortest exact "
        "span.\nQuestion: {q}\nAnswer:"
    )
    greedy = {"temperature": 0.0, "max_new_tokens": MAX_NEW}
    build_sp = {"temperature": 0.0, "max_new_tokens": 1}

    base_ttfts, rk_ttfts = [], []
    try:
        for idx, s in enumerate(samples):
            ctx = "\n\n".join(s["docs"])
            prompt = ctx + q_suffix.format(q=s["question"])

            # Baseline: dense full-context query TTFT.
            bt = _ttft_stream(engine, prompt, greedy, segments=None)

            # Build offline segments (NOT timed into the query TTFT).
            sids = []
            for di, d in enumerate(s["docs"]):
                dids = tok(d, add_special_tokens=False)["input_ids"]
                sid = OfflineKVCache.compute_segment_id(
                    MODEL, dids, prepend_bos=(di == 0)
                )
                engine.generate(
                    d, build_sp, redknot_offline_segments=["__RKBUILD__:" + sid]
                )
                sids.append(sid)

            # RedKnot: sparse query reusing offline segments, TTFT.
            rt = _ttft_stream(engine, prompt, greedy, segments=sids)

            base_ttfts.append(bt)
            rk_ttfts.append(rt)
            print(
                f"[ttft] sample {idx} ({s['ds']}): baseline={bt:.3f}s "
                f"redknot={rt:.3f}s speedup={bt / rt:.2f}x"
            )
    finally:
        engine.shutdown()

    if base_ttfts:
        import statistics as st

        b = st.mean(base_ttfts)
        r = st.mean(rk_ttfts)
        print("=" * 72)
        print(
            f" TTFT MEAN over {len(base_ttfts)} samples @ ctx={TARGET_TOKENS}: "
            f"baseline={b:.3f}s  redknot={r:.3f}s  speedup={b / r:.2f}x"
        )
        print("=" * 72)
    print("[ttft] DONE")


if __name__ == "__main__":
    main()
