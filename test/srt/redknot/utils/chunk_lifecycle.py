#!/usr/bin/env python3
"""Chunk reuse-lifecycle analysis for RedKnot offline-KV caching.

Question answered
-----------------
For a non-prefix document chunk (a passage that appears at *different*
positions across requests, so prefix-cache can never hit it), is it worth
keeping its KV in the offline cache?

Economics
---------
A cached chunk costs ``kv_bytes * residency_time`` of GPU memory and saves
``reuse_count * prefill_cost_per_chunk`` of recompute. So the value density is

    value = (reuse_count * prefill_saved) / (kv_bytes * residency)

This script measures, from a real multi-document QA stream (musique / KILT /
MS MARCO), for every chunk:
  * reuse_count      : how many requests reuse it
  * first_seen / last_seen (in request-index time) -> lifecycle span
  * residency        : last_seen - first_seen  (lifecycle length)
  * non_prefix_ratio : fraction of reuses where the chunk is NOT at position 0
                       (i.e. fraction prefix-cache would miss)

It then applies a caching policy: cache a chunk only if
``reuse_count >= R_MIN`` and ``residency <= T_MAX`` (reuse often, lifespan
short), and reports memory saved vs recompute saved vs a "cache everything"
and a "prefix-cache only" baseline.

Datasets supported
------------------
  * musique  (local jsonl, paragraphs[].title as stable chunk id)
  * kilt     (jsonl with provenance wikipedia_id)
  * msmarco  (jsonl with passage ids)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_musique_stream(path, max_q=None):
    """Yield (request_id, [chunk_id,...]) in arrival order.

    chunk_id = passage title (stable across questions). Position in the list
    = position in the request's context (used for non-prefix detection).
    """
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_q and i >= max_q:
                break
            r = json.loads(line)
            chunk_ids = []
            for para in r["paragraphs"]:
                chunk_ids.append(para["title"])
            rows.append(
                (r["id"], chunk_ids, [p["paragraph_text"] for p in r["paragraphs"]])
            )
    return rows


_PASSAGE_RE = None


def load_longbench_stream(path, max_q=None):
    """LongBench multi-doc format.

    context = "Passage 1:\\n<Title>\\n<text>\\n\\nPassage 2:\\n...".
    chunk_id = passage title (first non-empty line of each passage block).
    """
    import re

    global _PASSAGE_RE
    if _PASSAGE_RE is None:
        _PASSAGE_RE = re.compile(r"Passage\s+\d+:\s*")
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_q and i >= max_q:
                break
            r = json.loads(line)
            ctx = r.get("context", "")
            parts = [p.strip() for p in _PASSAGE_RE.split(ctx) if p.strip()]
            chunk_ids, texts = [], []
            for p in parts:
                title = p.split("\n", 1)[0].strip()
                if not title:
                    continue
                chunk_ids.append(title)
                texts.append(p)
            if chunk_ids:
                rows.append((r.get("_id", str(i)), chunk_ids, texts))
    return rows


def load_dureader_stream(path, max_q=None):
    """LongBench dureader (Chinese): blocks marked by '文章N' / '标题：'."""
    import re

    re_doc = re.compile(r"文章\s*\d+\s*")
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_q and i >= max_q:
                break
            r = json.loads(line)
            ctx = r.get("context", "")
            parts = [p.strip() for p in re_doc.split(ctx) if p.strip()]
            cids, texts = [], []
            for p in parts:
                first = p.split("\n", 1)[0].strip()
                title = first.replace("标题：", "").strip() or first
                cids.append(title)
                texts.append(p)
            if cids:
                rows.append((r.get("_id", str(i)), cids, texts))
    return rows


def load_stream(dataset, path, max_q=None):
    """Dispatch by dataset name / format."""
    if dataset in ("musique_ans", "musique_full"):
        return load_musique_stream(path, max_q)
    if dataset == "dureader":
        return load_dureader_stream(path, max_q)
    # LongBench tasks (hotpotqa, 2wikimqa, musique, ...)
    return load_longbench_stream(path, max_q)


def analyze(stream, tokenizer=None, kv_bytes_per_token=None):
    """Compute per-chunk lifecycle stats from an arrival-ordered stream."""
    first_seen = {}
    last_seen = {}
    reuse_count = defaultdict(int)
    nonprefix_hits = defaultdict(int)
    total_hits = defaultdict(int)
    chunk_text = {}

    for t, (rid, chunk_ids, texts) in enumerate(stream):
        for pos, (cid, txt) in enumerate(zip(chunk_ids, texts)):
            if cid not in first_seen:
                first_seen[cid] = t
                chunk_text[cid] = txt
            last_seen[cid] = t
            reuse_count[cid] += 1
            total_hits[cid] += 1
            if pos != 0:
                nonprefix_hits[cid] += 1

    stats = {}
    for cid in reuse_count:
        rc = reuse_count[cid]
        residency = last_seen[cid] - first_seen[cid]
        n_tok = (
            len(tokenizer(chunk_text[cid])["input_ids"])
            if tokenizer
            else len(chunk_text[cid].split())
        )
        stats[cid] = dict(
            reuse_count=rc,
            first_seen=first_seen[cid],
            last_seen=last_seen[cid],
            residency=residency,
            non_prefix_ratio=nonprefix_hits[cid] / max(total_hits[cid], 1),
            n_tokens=n_tok,
            kv_bytes=(n_tok * kv_bytes_per_token) if kv_bytes_per_token else n_tok,
        )
    return stats


def evaluate_policies(stats, r_min, t_max, n_requests):
    """Compare caching policies on memory cost vs recompute saved."""

    def summarize(name, keep_fn):
        kept = {c: s for c, s in stats.items() if keep_fn(s)}
        # recompute saved = (reuse_count - 1) prefills avoided per cached chunk
        saved_prefills = sum((s["reuse_count"] - 1) for s in kept.values())
        # memory-time cost = kv_bytes * residency (proxy GPU-memory*time)
        mem_time = sum(s["kv_bytes"] * max(s["residency"], 1) for s in kept.values())
        peak_mem = sum(s["kv_bytes"] for s in kept.values())
        nonprefix_saved = sum(
            (s["reuse_count"] - 1) * s["non_prefix_ratio"] for s in kept.values()
        )
        return dict(
            policy=name,
            chunks_cached=len(kept),
            prefills_saved=saved_prefills,
            nonprefix_prefills_saved=round(nonprefix_saved, 1),
            peak_kv_bytes=peak_mem,
            mem_time_cost=mem_time,
            value_density=round(saved_prefills / max(mem_time, 1) * 1e6, 3),
        )

    total = len(stats)
    rows = [
        summarize("cache_all", lambda s: True),
        summarize(
            "prefix_only",
            lambda s: s["non_prefix_ratio"] == 0.0 and s["reuse_count"] >= 2,
        ),
        summarize(
            f"redknot(R>={r_min},T<={t_max})",
            lambda s: s["reuse_count"] >= r_min and s["residency"] <= t_max,
        ),
    ]
    return rows, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique")
    ap.add_argument(
        "--path",
        default="/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl",
    )
    ap.add_argument("--max-q", type=int, default=None)
    ap.add_argument("--r-min", type=int, default=3, help="min reuse count to cache")
    ap.add_argument(
        "--t-max", type=int, default=500, help="max residency (requests) to cache"
    )
    ap.add_argument(
        "--model-path", default=None, help="tokenizer for real token counts"
    )
    ap.add_argument(
        "--kv-bytes-per-token",
        type=float,
        default=None,
        help="bytes of KV per token (e.g. 2*n_layer*n_kv_head*head_dim*2)",
    )
    ap.add_argument("--out", default=str(HERE / "figures/chunk_lifecycle_musique.json"))
    args = ap.parse_args()

    tok = None
    if args.model_path:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.dataset == "musique":
        stream = load_musique_stream(args.path, args.max_q)
    else:
        raise SystemExit(f"dataset {args.dataset} not wired yet")

    n_req = len(stream)
    stats = analyze(stream, tok, args.kv_bytes_per_token)

    # distributions
    reuse_counts = sorted((s["reuse_count"] for s in stats.values()), reverse=True)
    residencies = [s["residency"] for s in stats.values()]
    nonprefix = [s["non_prefix_ratio"] for s in stats.values()]
    reused2 = sum(1 for c in reuse_counts if c >= 2)

    print(f"\n=== Chunk reuse lifecycle: {args.dataset} ({n_req} requests) ===")
    print(f"unique chunks         : {len(stats)}")
    print(f"reused >=2            : {reused2} ({100 * reused2 / len(stats):.1f}%)")
    print(f"max reuse count       : {max(reuse_counts)}")
    print(f"mean reuse count      : {sum(reuse_counts) / len(reuse_counts):.2f}")
    print(f"mean residency (req)  : {sum(residencies) / len(residencies):.1f}")
    print(
        f"mean non-prefix ratio : {sum(nonprefix) / len(nonprefix):.3f}  "
        f"(fraction prefix-cache would MISS)"
    )

    rows, total = evaluate_policies(stats, args.r_min, args.t_max, n_req)
    print(f"\n=== Caching policy comparison (total chunks={total}) ===")
    hdr = f"{'policy':>26} | {'cached':>7} {'%cached':>7} | {'prefills_saved':>14} {'nonprefix_saved':>15} | {'peak_kv':>12} {'value_density':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['policy']:>26} | {r['chunks_cached']:>7} {100 * r['chunks_cached'] / total:>6.1f}% | "
            f"{r['prefills_saved']:>14} {r['nonprefix_prefills_saved']:>15} | "
            f"{r['peak_kv_bytes']:>12} {r['value_density']:>13}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "n_requests": n_req,
                "n_chunks": len(stats),
                "reused_ge2": reused2,
                "max_reuse": max(reuse_counts),
                "mean_reuse": sum(reuse_counts) / len(reuse_counts),
                "mean_residency": sum(residencies) / len(residencies),
                "mean_non_prefix_ratio": sum(nonprefix) / len(nonprefix),
                "policies": rows,
                "reuse_hist": reuse_counts[:200],
            },
            f,
            indent=2,
        )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
