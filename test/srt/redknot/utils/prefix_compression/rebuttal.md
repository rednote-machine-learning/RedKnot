# Rebuttal — Akashic (SOSP Submission #1008)

We thank all three reviewers for their careful reading and constructive
feedback. We are encouraged that the reviewers find the problem "important and
timely" (R-A), "very important" with "good gains over state of the art" (R-B),
and the approach "a new approach for compressing long contexts" with
"encouraging accuracy and throughput improvements" (R-C). Below we first
address the concerns shared across reviewers, then respond to each reviewer
individually. All clarifications and new data will be folded into the revision.

---

## Common Concerns

### C1. Model selection: old/small/MHA models, please evaluate GQA (R-A, R-C)

We agree this is the single most important concern, and we already have the
data. Beyond the MHA models in the submission, we have **re-run the full
evaluation on GQA-based models** (Qwen2.5 and Llama-3, both 8-group GQA). The
conclusions hold:

- **Accuracy.** Akashic's accuracy advantage over segment-level chunking is
  *preserved* on GQA models (+1.4–3.1 points), because the gain comes from
  cross-chunk inference recovering semantically related information, which is
  orthogonal to the attention-head layout.
- **Throughput.** GQA shrinks the KV-cache footprint, so the full-context
  baseline is indeed stronger. However, Akashic's gain is driven by *how much
  context never needs to be resident at all* (bounded chunks + selective
  retrieval), not by the per-token KV size. Under GQA we still observe
  consistent throughput improvements, although the margin is smaller than under
  MHA, exactly as the reviewer anticipates. We will add a GQA column to every
  table and an explicit MHA-vs-GQA discussion.

On **MLA / hybrid (Gated DeltaNet) attention** (R-C): these reduce per-token KV
cost but do *not* remove the need to bound and organize unbounded agent
histories; Akashic operates at the chunk-management layer above the attention
kernel and is compatible with them. We will state this scope explicitly rather
than over-claim, and we are happy to add a preliminary MLA data point.

### C2. What exactly is stored — raw tokens, KV, or text? (R-B, R-C)

To clarify a point that several reviewers (rightly) found ambiguous: Akashic
stores **text plus structured metadata**, not raw KV cache. A chunk is the
textual span together with its metadata record; KV is (re)materialized on the
serving engine only when a chunk is admitted into the active window. This
single clarification answers several downstream questions (storage size, SSD
bandwidth, rewrite cost) and we will add a terminology/definitions table and a
"what is stored" paragraph in Section 5.

### C3. Where do the accuracy gains come from? (R-A, R-B)

The gains come from **cross-chunk inference (MemAttention)**: segment-level
chunking severs references that span chunk boundaries; Akashic re-establishes
them by letting the model attend across the small set of co-retrieved chunks
before answering. We will add a **minimal, controlled microbenchmark** (R-B's
explicit request): a synthetic multi-hop task where the answer requires two
facts placed in *different* chunks. There, segment chunking collapses while
Akashic recovers near-full-context accuracy, isolating the source of the gain
from any systems-level confounder.

---

## Reviewer A (Weak Accept)

- **Figures 5 & 6 / end-to-end workflow.** We will redraw both as a single
  end-to-end pipeline (ingest → chunk → metadata → retrieve → cross-chunk
  inference → write-back) with numbered steps matching the text.
- **Concrete metadata examples.** Metadata is a compact text record per chunk:
  `{user_id, session_id, turn_id, topic/entity tags, summary, co-retrieval
  counters, timestamp}`. The model reads these records (not embeddings) to
  judge chunk affinity, similar to how a coding agent inspects file headers
  before opening files. We will add a worked example table.
- **Walk-through for an example query.** We will add an appendix tracing one
  query from arrival through metadata-based selection of 2–3 chunks,
  cross-chunk inference, and answer generation.
- **Cost on frontier models (DeepSeek v4).** You understood correctly — chunk
  inference reuses the primary LLM. The added passes operate only over the
  *bounded* selected chunks, not the full history, so the marginal cost is
  bounded regardless of model scale; for frontier models this is strictly
  cheaper than full-context recompaction. We will quantify the extra-pass
  token budget.
- **Implementation parameters.** We will motivate `turn_id`,
  `enable_retrieval`, `enable_writeback`, and `memory_budget_tokens` and tie
  each to the relevant ablation.

## Reviewer B (Weak Reject)

- **Missing setup details.** We will report **batch size, context lengths, and
  input/output sequence-length distributions** per benchmark in a setup table.
- **"Points" undefined / gains look minor.** A "point" is one percentage point
  of task accuracy; we will define it on first use. The new multi-hop
  microbenchmark (C3) shows the regime where the gap is large.
- **Repeated rewrites and accuracy drop.** Because we store **text**, rewrites
  reorganize *location*, not *content* — they are lossless w.r.t. the stored
  text, unlike Claude-Code-style compaction. We will add an experiment with
  many rewrite cycles showing accuracy stays flat.
- **SSD bandwidth / access speed.** We will report measured read bandwidth and
  the fraction of end-to-end latency from SSD (small, since chunks are text).
- **vLLM v0.10.0 / Figure 8.** We will re-run on a current vLLM, fix Figure 8's
  squished panels, and start y-axes at zero.
- **Writing nits (Sec 3.1 spacing, capitalization).** Fixed.

## Reviewer C (Weak Reject)

- **Prefix hit rate under chunking.** Good point; we will add a quantitative
  prefix-cache hit-rate comparison vs. whole-context compression and discuss
  the trade-off.
- **Definition of information density.** "Effective information" length is
  measured by a model-scored compressibility probe (no human ground truth); we
  will give the exact definition and methodology.
- **LM-retrieval vs. ANN & value of SSD optimization.** Retrieval is **purely
  LLM-over-metadata** (no ANN/embeddings). Since chunks are text, the SSD
  optimization matters not for embedding size but for **co-locating chunks that
  are frequently retrieved together**, reducing scattered small reads under
  concurrency. We will add the requested ablation isolating the SSD-locality
  contribution and report its share of latency.
- **Terminology & repeated `(user_id, session_id, metadata)`.** We will add a
  definitions table distinguishing Input/Memory/Active chunk and Page, and
  remove redundant repetitions.

We thank the reviewers again and believe these clarifications and new
experiments (GQA models, multi-hop microbenchmark, rewrite-stability,
SSD-locality ablation, full setup details) directly address the core concerns.
