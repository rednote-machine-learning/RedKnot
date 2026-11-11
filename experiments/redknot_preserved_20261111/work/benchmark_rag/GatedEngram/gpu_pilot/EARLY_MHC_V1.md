# Early-six sensitivity + per-layer mHC pilot

Independent experimental module. Does not change v8, the formal Engram gate,
or any paused experiment. Full/global source layers (including SWA) execute
normally; their numerical values may still reflect hybrid upstream hidden.

For each native 8192-token chunk, layers 0..5 execute fully. Each token receives
the mean of six SAME-layer online-versus-independent-offline canonical SWA KV
relative L2 errors. Computation uses postnorm/preRoPE BF16 KV and FP64 reduction.
This is not adjacent-layer similarity and not an accuracy certificate.

After layer 5, select the highest-error 20% or 50% of eligible interior rows.
Tie-breaking is input position. Mandatory rows are additive: question/template,
first/last 128 of each independent 32768-token cache document, and first/last
128 of every native chunk. Keep this mask through ordinary layers
6,7,9,10,11,12,13,15,16,17,18. All other layers execute normally.

Selected rows run actual attention projections, attention, output projection,
and complete FFN/MoE. Unselected SWA KV comes from the immutable same-layer bank
and receives target-position RoPE/native FP8 cast. Unselected attention and FFN
outputs are zero, but native full-row mHC pre/post and normalization still run
at EVERY layer. Neither old hidden nor offline FFN output is substituted. This
is different from the previous interval-frozen residual carrier experiment.

First run a 16384-token, two-native-chunk all-selected operator qualification:
compare every decoder output and all four lazy mHC fields by exact byte SHA
across all40 layers, and check final output IDs. No threshold relaxation or
automatic retry. This is a bounded operator qualification, not a full256K
numerical-equivalence claim or an accuracy result.

The first GPU pilot uses the already-exposed HotpotQA case2 and 2WikiMQA case3,
one each, at actual 8x32768 context tokens plus template/question. These are
LongBench-expanded inputs, not native256K questions or independent new holdout.
Prior full runs answered both incorrectly: equal EM0 on this tiny pilot cannot
establish preservation of accuracy. Keep all zero scores and regressions.

Three arms: same-run native full, early20, early50. Quality EOS cap64, EM/F1
against fixed gold, keep truncations. C1 TTFT: one warmup/arm and three formal
32-token runs/arm, rotating order. Include online normalization for scores,
ranking, planning, gather/scatter, cache restore and mHC in TTFT. Exclude model
loading, offline cache building, and untimed qualification; report offline cost
separately. No C4, saturation or whole-model FLOP claim. Actual skip/reuse rates
refer only to the eleven eligible layers.

CPU tests and check-only precede a new bounded original-supervisor launch. The
original identity-verified holder must physically cover all eight GPUs before
launch and after terminal cleanup. Freeze all source/input/protocol/reader
files once the task is launched. Preserve raw phases, per-token scores, actual
selected masks, offline payload identities, exact qualification fingerprints,
and complete gold/answer/length evidence for independent auditing.
