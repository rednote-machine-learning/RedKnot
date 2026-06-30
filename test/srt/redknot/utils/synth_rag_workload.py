#!/usr/bin/env python3
"""Synthesize realistic RAG query streams from a shared passage pool.

Real production RAG retrieval follows a skewed (Zipfian) popularity: a small
set of documents is retrieved by many queries, the long tail rarely. Public
QA slices (200 questions) are too small to exhibit this, so we ground a
controllable workload in the *real* musique passage pool and sample retrievals
with a tunable Zipf exponent.

This yields several "datasets" (different skew) all using real passages, so the
LRU cache study has multiple reuse regimes to validate against.
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUSIQUE = "/mnt/tidal-alsh01/dataset/redone/096/RedCacheV0.2/datasets/musique_ans_v1.0_dev.jsonl"


def build_pool(path, max_passages=20000):
    """Collect unique (title -> text) passages from musique_ans."""
    pool = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            for p in r["paragraphs"]:
                t = p["title"]
                if t not in pool:
                    pool[t] = p["paragraph_text"]
                if len(pool) >= max_passages:
                    return pool
    return pool


def gen_stream(pool_titles, n_queries, k_per_query, zipf_s, seed=0):
    """Generate (qid, [titles], [texts]) with Zipf-popular passage retrieval."""
    rng = random.Random(seed)
    N = len(pool_titles)
    # Zipf weights over the ranked passage pool
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(N)]
    total = sum(weights)
    weights = [w / total for w in weights]
    # precompute cumulative for sampling
    import bisect

    cum = []
    acc = 0.0
    for w in weights:
        acc += w
        cum.append(acc)

    def sample_one():
        x = rng.random()
        return bisect.bisect_left(cum, x)

    stream = []
    for q in range(n_queries):
        picked = set()
        while len(picked) < k_per_query:
            picked.add(sample_one())
        idxs = list(picked)
        rng.shuffle(idxs)  # non-prefix: random positions
        titles = [pool_titles[i] for i in idxs]
        stream.append((f"q{q}", titles, titles))
    return stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10, help="passages retrieved per query")
    ap.add_argument(
        "--zipf", type=float, default=1.1, help="Zipf exponent (higher=more skew)"
    )
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = build_pool(MUSIQUE, args.pool)
    titles = list(pool.keys())
    stream = gen_stream(titles, args.n_queries, args.k, args.zipf, args.seed)

    # quick reuse stat
    from collections import Counter

    c = Counter()
    for _, ts, _ in stream:
        for t in set(ts):
            c[t] += 1
    reused = sum(1 for v in c.values() if v >= 2)
    print(
        f"synth: zipf={args.zipf} queries={args.n_queries} k={args.k} pool={len(titles)}"
    )
    print(
        f"  touched passages={len(c)} reused>=2={reused} ({100 * reused / max(len(c), 1):.0f}%) "
        f"max_reuse={max(c.values())}"
    )

    out = args.out or str(HERE / f"figures/synth_zipf{args.zipf}.jsonl")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for qid, ts, _ in stream:
            # store as LongBench-like: paragraphs with title+text
            paras = [{"title": t, "paragraph_text": pool[t]} for t in ts]
            f.write(json.dumps({"id": qid, "paragraphs": paras}) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
