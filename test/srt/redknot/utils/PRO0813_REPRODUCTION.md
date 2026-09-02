# DeepSeek-V4-Pro-0813 isolated reproduction

This directory keeps the Pro-0813 benchmark separate from every Flash-0731
entry point. The production run is pinned to:

- `/workspace/Models/DeepSeek-V4-Pro-0813`
- `server/start_server_redknot_pro0813.sh`
- 8x NVIDIA B300 / Blackwell SM103, TP8
- 61 layers and 128 logical attention heads
- dense layers `0,1,2,58,59,60`; reusable layers `3..57`
- Indexer Top-K 1024
- Pro `3_55_3` execution profiles

The certified runtime is PyTorch `2.9.1+cu129` with CUDA 12.9.  Its shared
`sgl_kernel 0.3.20` wheel does not contain an SM103 RMSNorm image, so the Pro
launcher requires `SGLANG_USE_JIT_RMSNORM=1` and refuses an explicit override.
The environment's pip NVRTC is 12.8 and does not recognize `compute_103`;
PyTorch's embedded RPATH resolves it ahead of `LD_LIBRARY_PATH`. The launcher
and standalone holder guard therefore preload the system CUDA 12.9
`libnvrtc.so.12` before `libnvJitLink.so.12`. This is required for PyTorch
Jiterator operations on B300 and is isolated to the Pro process environment.
Before model workers start it verifies exactly eight `B300` devices at compute
capability `10.3`, then runs
`probe_pro0813_jit_rmsnorm_sm103.py` against BF16 widths 512, 1536 and 7168 in
both ordinary and fused-residual forms.  A static `B300` label is not accepted
as hardware evidence.

FlashInfer 0.5.3 must compile B300 kernels for `10.3a`; a bare explicit `10.3`
omits the architecture feature suffix and makes its MXFP8 E2M1 instruction fail
in `ptxas`. The launcher therefore rejects a conflicting
`FLASHINFER_CUDA_ARCH_LIST` and pins `10.3a`. It also runs
`probe_pro0813_triton_h1_sm103.py`, which constructs the production 584-byte
DSV4 FP8/BF16 page layout and independently checks the exact fused headwise
single- and dual-scope Triton path, including per-head scope masks and MAIN
lengths, against a PyTorch reference. The standalone
`run_sm103_probe_with_holder_guard.sh` guard also pins
`SGLANG_USE_JIT_RMSNORM=1` and `FLASHINFER_CUDA_ARCH_LIST=10.3a`, runs both of
these startup oracles before the remaining SM103 probes, and passes
`--expected-source-root /workspace/RedKnot/python` to the adversarial
Triton probe. Thus a probe cannot silently import the Flash tree or an
unrelated installed package.

CPU-only validation (does not import torch or touch a GPU):

```bash
python test/srt/redknot/benchmark_dsv4_pro0813_redknot_http.py \
  --contract-only \
  --contract-config test/srt/redknot/deepseek_v4_pro0813_config.json
```

The exact BF16 z_off capacity per TP rank is:

```text
55 reusable layers * total_tokens * 2 output groups/rank * 1024 * 2 bytes
```

At 64K this is `14,763,950,080` bytes per rank. The default 64K cap is
16 GiB per rank.

After the official model manifest is complete and GPU kernel probes pass, run
one formal target. Passing the token count first selects a target-derived,
timestamped `RUN_DIR` automatically:

```bash
test/srt/redknot/run_deepseek_v4_pro0813_reproduction.sh 65536
```

The names are `pro0813-{64k,128k,256k,440k,512k}-TIMESTAMP`. The historical
path-first form remains available when an exact output path is required:

```bash
test/srt/redknot/run_deepseek_v4_pro0813_reproduction.sh \
  /workspace/RedKnot/results/formal-64k 65536
```

Formal execution is fail-closed by default: TTFT warmup/measurement counts are
`3/10`, QPS warmup/measurement waves are `3/10`, QPS measurement is enabled,
the certified first-document-prefix QPS concurrency is `1`, quality repeats
are `3`, the row-sparse active ratio is `0.20`, and the supervisor passes
`--strict-performance` explicitly. Sampling below
those minima is rejected before the holder handoff. A short claim-ineligible
diagnostic must opt out explicitly, for example:

```bash
REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1 \
REDKNOT_QPS_WARMUP_WAVES=1 REDKNOT_QPS_WAVES=2 \
test/srt/redknot/run_deepseek_v4_pro0813_reproduction.sh 65536
```

The supervisor selects `--combined-headsplit-row-sparse`, which is the
full-composite shared-restore + sparse-Q + fused-z path, in both formal mode
and a performance-only diagnostic opt-out. Thus
`REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1` changes claim eligibility without
silently changing the algorithm under test. To run the legacy zoff-only
attribution arm, set both `REDKNOT_PRO0813_DIAGNOSTIC_PERFORMANCE=1` and
`REDKNOT_PRO0813_DIAGNOSTIC_ZOFF_ONLY=1`; only then does the supervisor select
`--combined-headsplit-row-sparse-diagnostic-zoff-only`. Strict mode refuses to
treat either diagnostic record as a successful formal run even if all
diagnostic execution checks complete.

To run one formal case at every supported length in the required order, use
the dedicated sequencer. It binds these canonical frozen single profiles:

- `test/srt/redknot/qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json`
- `test/srt/redknot/qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json`

```bash
test/srt/redknot/utils/run_deepseek_v4_pro0813_all_targets.sh
```

The sequencer does not accept alternate 440K/512K profile paths. If
`REDKNOT_QUALIFICATION_PROFILE_440K` or
`REDKNOT_QUALIFICATION_PROFILE_512K` is present, it must equal the canonical
path above; both profiles must also match the fixed SHA-256 values below. Both
assets are verified before the 64K runner is invoked, so a missing or replaced
long-run asset fails before any target, model, holder, or GPU server action.
The stdlib verifier binds each profile to its exact target, official tokenizer,
raw dataset, selected rows, prompt artifacts, geometry, full-combined intent,
sidecar digest, and co-located non-symlink files. Its profile SHA-256 is written
to `sequence.plan.tsv` and passed through every later verification boundary.

All five target invocations receive the same sanitized formal environment. In
addition to TTFT `3/10`, QPS concurrency `1` and waves `3/10`, quality repeats
`3`, and row-sparse ratio `0.20`, the sequencer pins the adaptive controller,
combined algorithm, adaptive-TopK policy, deterministic seed, FlashInfer MXFP4
backend, TileLang/DeepGEMM/CUBLAS fast-path selectors, JIT RMSNorm, and
FlashInfer `10.3a` architecture contract. It unsets ambient head configuration,
SWA ratio, server-policy output, and server nonce values. A caller's inherited
experimental settings therefore cannot change one member of the formal sweep.

This sequence is exactly `64K -> 128K -> 256K -> 440K -> 512K`, with one
profile per target. It deliberately does not expand any 15-case release suite;
suite reproduction remains a separate explicit workload.

The supervisor authenticates the exact no-argument `gpu_hold.py` process and
its eight workers by process group, physical GPU index, GPU UUID, and PID. A
holder is considered utilization-ready only after at least three of at most
fifteen one-second samples show every B300 at or above 90%; an instantaneous
sample or an exact 100% requirement is not used. It then swaps to the exact
low-memory bootstrap holder during TP worker import. During the barrier it
rechecks the bootstrap process identity every second and its eight-UUID worker
coverage every ten seconds. Every compute-app or inventory query used to
decide ownership, coverage, utilization, or idleness is retried at most three
times, and a query or schema failure is never interpreted as an idle GPU.

At the barrier, the supervisor reauthenticates the holder immediately before
stopping it, proves no unexpected GPU PID remains, and publishes the go record
as a same-directory, no-replace atomic link. It verifies the public path is the
same regular inode with exact content before accepting release, then proves the
eight TP worker PIDs cover all eight physical UUIDs. EXIT cleanup first stops
the authenticated server and waits for proven GPU idleness; only then does it
start and verify the full holder with the same coverage and utilization gates.
If telemetry cannot prove idleness or restoration health, cleanup reports a
failure instead of starting an overlapping holder. The immutable official-model
gate runs before holder discovery or handoff. All of these entry points live
only in the Pro staging tree and do not modify the Flash reproduction path.

## GPU evidence required for a performance claim

- The B300 JIT RMSNorm and adversarial production `triton_h1` numerical oracles
  run in both the certified server launcher and the standalone holder guard;
  the full model result must still carry its actual hardware/runtime receipts.
- The B300/SM103 FlashMLA code object has a separate numerical oracle in the
  standalone holder guard.
- The Pro TP8 `o_groups/rank=2` z_off merge path needs its B300 numerical oracle
  before any performance claim.
- 440K/512K runs require a frozen qualification profile; the wrapper rejects
  those lengths without `REDKNOT_QUALIFICATION_PROFILE`.
- 64K is expected to be the minimum-benefit point, but that remains an
  experimental hypothesis until dense/reuse quality, TTFT, and QPS all pass.

## Canonical 440K/512K qualification inputs

The formal Pro profiles are independent, output-blind HotpotQA 10-query
cohorts.  They use co-located relative data/prompt artifacts and fixed profile
sidecars; stale `/mnt/tidal-alsh01/...` profiles and the Flash 15-case suite are
not valid inputs to the Pro single-profile runner.

```text
440K: test/srt/redknot/qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json
      sha256 52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b
512K: test/srt/redknot/qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json
      sha256 e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d
```

Before any GPU action, run
`verify_pro0813_qualification_profile.py PROFILE --expected-target-tokens N`.
The HTTP consumer performs the same fail-closed verification and resolves only
same-directory relative artifacts.  See `qualification_profiles/README.md`
for the exact rebuild commands and complete hash/quota contract.

These profiles declare `full_combined_production_v1` as the intended formal
execution profile.  A legacy `zoff_only` run is diagnostic and cannot be used
as the formal 440K/512K result.
