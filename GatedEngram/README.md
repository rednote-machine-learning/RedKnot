# KV Cache Engram: adjacent-SWA / FFN-only experiment

This is an **opt-in research implementation**, not the default DeepInfer forward
path or an accuracy-qualified cache admission policy. Importing the package does
not import Torch, initialize CUDA, install hooks, or start a model. No GPU
accuracy, TTFT improvement, or Pareto-optimality is established by publishing it.

## Precisely what this version does

Layer numbers below are zero-based unless explicitly labelled otherwise.

1. Layers 0–5 use full native computation. At layers 2–5 (human layers 3–6),
   capture each layer's own SWA KV after its native normalization and before
   RoPE. Global KV is not the selector feature: these layers can share the same
   global KV producer, making their apparent global-KV equality structural.
2. For each token, compute the arithmetic mean of three relative L2 changes:
   `||KV[l+1] - KV[l]||2 / max(||KV[l]||2, 1e-8)`, for `l = 2, 3, 4`.
   Rank **tokens**, not layer averages. Within each native 8192-token chunk,
   select `ceil(15 * document_token_count / 100)` document tokens; ties use
   ascending input position. Template/question tokens are additionally retained.
3. After layer 5, all native Q, attention and output projections still execute.
   Global KV/index/compressor updates are not skipped. SWA KV instead uses an
   authenticated, independent-document offline bank, restored to the current
   positions and native cache cast. Only the first **127** tokens of each actual
   document and non-document template/question tokens require online SWA KV
   projection. There is no hidden document-tail or native-8K-boundary guard.
4. The FFN/MoE mask is **independent of** the SWA projection mask. Selected rows
   execute the original FFN/MoE; other present rows contribute a zero FFN branch.
   Native mHC/norm/residual processing remains in place. No cached hidden state,
   FFN output, or identity replacement for mHC is used.
5. Native YOCO compaction is retained online. After its boundary, the FFN mask
   is intersected with the native surviving query positions. Tokens removed by
   native YOCO are **not forced through** later layers. Do not claim that all
   originally selected document tokens reach the final layer.

The earlier joint Q/attention/FFN token-mask experiment is not this algorithm.
The compatibility module `early_mhc_adapter.py` supplies packed-FP8 row-gather
utilities; its old joint-sparse adapter is not installed by this controller.

## Offline and online costs are different

Each independent 32768-token document needs real per-layer offline KV. For the
current target checkpoint, upper-layer SWA projections use their own layer
hidden states rather than a single shared SWA hidden state. Offline generation
therefore temporarily disables the model's YOCO **layout compaction** for that
independent-document request, computes all positions through all 40 layers,
and restores the original boundary before online evaluation or cleanup. It
does not change checkpoint weights, native source files, or the online YOCO
policy. This request-scoped layout override requires exclusive model ownership.

An eight-document bank with 40 layers, 32768 rows, 512 BF16 values per row is
10 GiB. Offline build time and storage must be reported separately. Online TTFT
must include scoring, ranking, index preparation, projection splitting,
gather/scatter, target-position KV restoration, mHC and synchronization. It
excludes model loading and prior offline construction, so it is **not cold-start
end-to-end latency**. Cache reuse fractions are KV-row fractions, not whole-model
FLOP reductions.

## Runtime boundary and safety

`kv_cache_engram.ffn_only.FFNOnlyController` is an isolated instance adapter for
the explicitly checked 40-layer DeepSeek-V4.1-Flash geometry on an external
DeepInfer SM90/CP1 runtime. This public repository's stock hardware support does
not by itself provide that SM90 runtime. An application must supply its existing
model, immutable token/document identities, execution identity, and a canonical
request binding containing `request_id`, `phase`, `start`, and `end`. It must
serialize requests while hooks are installed. This module does not ship private
SSH settings, checkpoint paths, production credentials, or a GPU scheduler.

The intended lifecycle is install → new native/all-selected numerical
qualification → independent offline capture → exact eight-document online
request → finish → close. All-selected qualification compares all decoder and
mHC output bytes across the declared two-chunk input; split Q/SWA and packed-row
gather checks are additional requirements. It is not a 256K accuracy guarantee.
CPU tests cannot substitute for this native numerical qualification or the full
length partial-path smoke test.

Invalid cache identity/content/version, incompatible shapes, stale plans,
nonfinite scores, changed hook ownership and incomplete phases fail closed.
There is no automatic retry after a stateful forward error. An empty local MoE
selection must never silently bypass EP collectives; unsupported empty-row
execution is rejected. Boundary refresh is not proof that every reused deep KV
equals the full-context value. Adjacent-layer changes also involve different
learned projection spaces and are only a ranking heuristic, **not a certified
reuse admission test**. The existing production admission requirements are not
relaxed by this diagnostic implementation.

## CPU checks

From this directory, run the independent CPU test files:

```bash
python -m unittest discover -s tests -p 'test_adjacent_kv_selector.py' -v
python -m unittest discover -s tests -p 'test_ffn_only*.py' -v
```

The selector and mask tests use the Python standard library. Tensor ABI and
controller tests additionally require Torch and must run on CPU with CUDA
initialization blocked. Any skipped tensor tests must be reported rather than
described as full native validation. There is no model download or GPU launch
in these commands.

## Evaluation contract

The independent pilot compares same-run full native computation with this
single FFN15 candidate on matched HotpotQA and 2WikiMQA examples, using eight
32K independent blocks per example. These are expanded LongBench materials,
not native 256K questions. Previously inspected exploratory examples are not a
new independent holdout. Calibration examples must not enter the quality
denominator.

Keep complete gold/answers, EOS/length truncation, per-token transition scores,
ranked masks, SWA online/reused positions, actual FFN and compact-layout row
counts, and immutable raw evidence. Compare EM and token F1 against gold, not
just against the baseline's text. Timed C1 uses the same fixed output length,
warmups and paired alternating repeats for both arms. Only closed paired
quality and timing results enter summaries. Preserve zero scores and slowdowns.

For each quality metric, a point dominates another only if quality is no worse,
TTFT is no higher, and at least one is strictly better. Identical points remain.
This gives at most a **two-point empirical frontier**, not a globally optimal
retention rate, statistical significance, or a claim about untested candidates.
