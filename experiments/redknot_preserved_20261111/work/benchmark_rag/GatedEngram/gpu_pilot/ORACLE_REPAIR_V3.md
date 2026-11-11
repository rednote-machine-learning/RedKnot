# SWA KV controlled reuse and oracle repair — v3

This is the separately authorized `kce-oracle-repair-20261102-v3` diagnostic.
It does not alter v2, v8, any paused experiment, or the formal admission gate.

## Question and arms

On the same four already-exposed v2 inputs (HotpotQA indices 2/4, 2WikiMQA 3/5),
compare fixed-gold EM/F1 after full downstream propagation:

| Arm | Prepared SWA KV used by attention |
| --- | --- |
| A / `recomputed` | Native full KV; capture answer-blind repair references |
| B / `swa_reuse` | Force offline KV into the prescribed candidate rows |
| C / `swa_reuse_repair` | Same candidates as B, restore A's chosen block per layer/chunk |

Each context remains eight independent 32768-token documents (262144 context
tokens plus template/question). These LongBench-derived expanded inputs are
not native 256K questions and are not a fresh independent holdout. The old two
calibration questions do not enter the four-question quality denominator.

## Intervention and RoPE

Only layers 6,7,9,10,11,12,13,16,17,18 are intervened. Layers 0..5 and 15 remain
full, as do all other noncandidate layers. Every Q, KV projection, attention,
hidden update, FFN/MoE, native global KV/index path still executes.

Offline canonical SWA KV is post-normalization/pre-RoPE. Candidate rows are
transformed to their current absolute target positions, with the native
RoPE and FP8 cast/back path, before attention consumes them. C inserts A's
already prepared post-RoPE/cast KV values directly: it does not rotate twice.

Both B and C preserve each document's first/last128 rows, template/question,
source-block fragments crossing native8192 chunks, and the last128 rows of
each native chunk. The additional carry-tail guard prevents a private
current-chunk intervention from silently disagreeing with the native SWA
carry store. The controller returns the intervened attention context, so later
layers and FFNs really consume the change; it is not a local observer only.

## Frozen A-derived repair

A compares native prepared KV against the target-position-restored offline
KV, scores each eligible512-token source block by maximum per-row relativeL2
(norm floor1e-8), and picks the highest-error source block per layer/native
chunk; ties use the earliest input position. Some protected boundaries shorten
the eligible part of a block. All candidate geometry and ranking scores are
sealed. Gold and generated answers are not inputs to this selection.

The exact selected A rows are stored in a bounded immutable CPU bank, with
shape/dtype/version/byte-digest/coverage checks. The plan freezes before B.
B and C share all candidate positions and the plan. C restores only that
selected source block. A is a full-computation oracle, so this diagnoses
recoverability; it is not a deployable low-cost online repair selector.
Inserting clean-A KV into hybrid-Q/hidden may help or hurt; neither outcome
is assumed and negative effects are retained.

## Measurement and acceptance

Natural EOS, max64 tokens, same fixed gold for all arms. Report complete raw
three-arm closures only, per dataset and overall: A→B loss, B→C gain, A→C
remaining loss (EM/F1 percentage points), wrong→correct/correct→wrong,
full answers/gold/truncation and actual source/injected/restored/remaining
cache row counts. Gap recovery `(C-B)/(A-B)` applies only when A>B; do not clip
negative or greater-than-one recovery. If A≤B, report not applicable.

This version computes all native projections and stores A oracle references.
It does **not** measure or claim TTFT acceleration, skipped FLOPs, production
admission, statistical significance, or generalization. Phase wall time and
offline construction time are diagnostic costs only. Forced substitutions
that did not pass the gate are authorized only in this diagnostic.

## Serial lifecycle

`launch_repair_pilot.py` starts a finite CPU-only wait, not a parallel GPU run.
The predecessor must have a successful original supervisor exit and physical
holder audit, all predecessor processes must exit, and the SHA-pinned v2 raw
reader must audit4/4 before a fresh eight-card holder identity/UUID/positive
occupancy check. Only the original supervisor releases/restores the holder.
All outputs are exclusive, source/input hashes frozen, no automatic retry.
Any predecessor failure blocks this queue without stopping unknown processes.

The new GPU run repeats exact split/gather/post-RoPE numerical qualification.
Failure aborts; no threshold relaxation. On final success or failure, require
both supervisor restoration and actual eight-GPU original-holder verification.

## Entry points

- `repair_protocol.py`: fixed independent protocol and raw validation contracts.
- `run_repair_pilot.py`: fresh qualification, offline build, A/B/C rollout.
- `kv_cache_engram/oracle_repair.py`: private prepared-KV intervention controller.
- `read_repair_results.py`: independent read-only sealed-evidence scoring.
- `launch_repair_pilot.py`: finite serial queue and original-holder handoff.
- `validate_cpu.py`: CPU regression suite with accelerator entry points blocked.
