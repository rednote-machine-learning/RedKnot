# DeepSeek V4 Flash + RedKnot release benchmark

The shortest one-command path is:

```bash
cd /mnt/tidal-alsh01/dataset/redone/RedKnot_Deepseekv4/test/srt/redknot
./run_deepseek_v4_flash_reproduction.sh
```

The wrapper creates the pinned environment only when it is absent, validates
an existing environment, sources it, and forwards any extra CLI arguments to
`benchmark_RedKnot_DeepSeekV4Flash.py`. It also holds a single-instance lock;
launching a second TP8 benchmark while one is active exits with a concise
message instead of stopping or corrupting the first run.

With no extra arguments, both the wrapper and Python entrypoint run four frozen
suites in order:

1. `datasets/LongBench/suites/release_64k_15case.jsonl`
2. `datasets/LongBench/suites/release_128k_15case.jsonl`
3. `datasets/LongBench/suites/release_256k_15case.jsonl`
4. `datasets/LongBench/suites/release_440k_15case.jsonl`

`datasets/LongBench/suites/RELEASE_SUITES.json` freezes their order, SHA256
digests and hot-state execution contract. The one-command entrypoint validates
that manifest before it starts any child suite, so edited or mismatched JSONL
files fail before GPU execution.

Each suite contains ten short-answer cases followed by five 30-token
long-answer cases. Each per-length `comparison.md` prints exactly fifteen
complete Recomputed/RedKnot output pairs in JSONL order. Recomputed means the
same DeepSeek-V4-Flash checkpoint performs a full online recomputation without
RedKnot reuse; it does not mean that the MoE model becomes a dense model.
Long-output presentation
cases are explicitly excluded from the primary short-answer accuracy
aggregate. A parent `four_length_summary.json` records all four immutable suite
hashes, output directories and exit codes.

The default TTFT protocol is hot-state: each frozen case runs three unmeasured
warmup pairs followed by ten measured Recomputed/RedKnot pairs, and reports
p50/p95. Offline snapshot construction, model loading and first-use kernel
compilation are not counted as online TTFT. Each case also generates one
complete text pair for the side-by-side output report.

Enter the release directory first. Install once, then source the pinned runtime
before every new shell session:

```bash
cd /mnt/tidal-alsh01/dataset/redone/RedKnot_Deepseekv4/test/srt/redknot
./setup_deepseek_v4_flash_env.sh
source ./environment-deepseek-v4-flash.env
python benchmark_RedKnot_DeepSeekV4Flash.py
```

The setup script creates `.venv_tf5`, installs the pinned CUDA 12.8/CPython
3.11 runtime from `requirements.txt`, installs this repository in editable
mode without replacing its dependency lock, and verifies the exact Torch,
FlashMLA, DeepGEMM, SGL Kernel and tokenizer APIs. To audit an existing
environment without changing it:

```bash
./setup_deepseek_v4_flash_env.sh --check-only
```

The full setup command is only needed when creating or repairing the virtual
environment. On an already prepared machine, `--check-only` followed by
`source` is sufficient. Sourcing the environment now also activates the pinned
virtual environment on `PATH`, so `python` and `$REDKNOT_PYTHON` resolve to the
same executable.

The first exact 16K/32K/55K snapshot on a cold machine can spend several minutes
in host-side validation and CUDA/Triton compilation. This is expected offline
preparation, not online TTFT and not a deadlock. The launcher mirrors
`driver.log` to the terminal and prints a 30-second heartbeat while the
benchmark is alive. Do not interrupt it merely because GPU utilization is low
during that first cold snapshot.

The optional 5-dataset × 10-case × 2-length matrix is intentionally long. Run it
in `tmux` (or another persistent terminal) and use `--resume` after a genuine
machine interruption:

```bash
tmux new -s redknot-release
cd /mnt/tidal-alsh01/dataset/redone/RedKnot_Deepseekv4/test/srt/redknot
./setup_deepseek_v4_flash_env.sh --check-only
source ./environment-deepseek-v4-flash.env
python benchmark_RedKnot_DeepSeekV4Flash.py \
  --datasets hotpotqa,2wikimqa,musique,multifieldqa_en,triviaqa \
  --lengths 256K,440K --cases-per-dataset 10 --resume \
  --output-dir /mnt/tidal-alsh01/dataset/redone/redknot_runs/release-reproduction
```

The validated machine used CPython 3.11.13, PyTorch 2.9.1 with CUDA 12.8,
FlashMLA 1.0.0+9241ae3, SGL Kernel 0.3.20, and NVIDIA driver 570.148.08 on
eight NVIDIA H200 GPUs. Hardware-specific kernels must be rebuilt when the
Python, CUDA or GPU ABI differs.

The DeepSeek-V4 runtime and benchmark implementation are copied directly from
RedKnotV0.1; this release entrypoint does not reimplement the algorithm. The
default reproduction is the packaged 64K, 128K, 256K and 440K 15-case suites; an explicit
matrix can cover five packaged LongBench datasets at both lengths. Before
touching a GPU it verifies model topology, every dataset SHA256, the published
43×64 head-policy reference, sparse-MoE policy, and each frozen profile. The
runtime deliberately keeps V0.1's built-in stride-8 head classifier; the JSON
under `head_class/` is its byte-auditable publication form, not a second policy.
It preserves the GPU holder during CPU preparation, releases it only for the
model run, and restores an eight-worker full-memory holder after success,
benchmark gate failure, crash, or signal.

Useful bounded runs:

```bash
# Validate any frozen 15-case suite; no GPU use.
python benchmark_RedKnot_DeepSeekV4Flash.py --prepare-only \
  --suite-jsonl datasets/LongBench/suites/release_64k_15case.jsonl
python benchmark_RedKnot_DeepSeekV4Flash.py --prepare-only \
  --suite-jsonl datasets/LongBench/suites/release_128k_15case.jsonl
python benchmark_RedKnot_DeepSeekV4Flash.py --prepare-only \
  --suite-jsonl datasets/LongBench/suites/release_256k_15case.jsonl
python benchmark_RedKnot_DeepSeekV4Flash.py --prepare-only \
  --suite-jsonl datasets/LongBench/suites/release_440k_15case.jsonl

# Validate/download inputs and build the explicit 256K/440K matrix cohorts.
python benchmark_RedKnot_DeepSeekV4Flash.py --prepare-only \
  --datasets hotpotqa,2wikimqa,musique,multifieldqa_en,triviaqa \
  --lengths 256K,440K --cases-per-dataset 10

# Resume an interrupted matrix without rerunning completed result.json files.
python benchmark_RedKnot_DeepSeekV4Flash.py --resume --output-dir /path/to/run

# One 256K cohort.
python benchmark_RedKnot_DeepSeekV4Flash.py \
  --datasets musique --lengths 256K

# Override the number of distinct frozen inputs per dataset.
python benchmark_RedKnot_DeepSeekV4Flash.py \
  --datasets hotpotqa,musique,triviaqa --lengths 256K \
  --cases-per-dataset 10

# Supplemental long-output matrix. Explicit matrix runs without this flag stay
# byte-for-byte on the shortest-span prompt/profile path. Long-output profiles
# live in separate long30/long50 cohort directories and results show the actual
# Recomputed/RedKnot generated-token counts beside the complete text.
./run_deepseek_v4_flash_reproduction.sh \
  --datasets hotpotqa,musique,multifieldqa_en --lengths 256K,440K \
  --cases-per-dataset 3 --long-output-tokens 30
./run_deepseek_v4_flash_reproduction.sh \
  --datasets hotpotqa,musique,multifieldqa_en --lengths 256K,440K \
  --cases-per-dataset 3 --long-output-tokens 50

# Reproduce the published long-output showcase with one command. This reruns
# the complete 3-dataset source pool, including non-selected cases; the five
# post-hoc presentation examples per length remain excluded from accuracy
# aggregates and are stored under datasets/LongBench/showcase/.
./run_deepseek_v4_flash_showcase_reproduction.sh

# Exact V0.1 256K MuSiQue reference (3 warmups, 5 TTFT pairs, 2 full outputs).
python benchmark_RedKnot_DeepSeekV4Flash.py --v01-reference

# One 440K cohort with a small single-flight QPS diagnostic.
python benchmark_RedKnot_DeepSeekV4Flash.py \
  --datasets musique --lengths 440K --measure-qps \
  --qps-concurrencies 1
```

The default checkpoint path is
`/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4-Flash-0731`.
Use `--model-path` or `REDKNOT_MODEL_PATH` to override it. If the checkpoint is
missing, the entrypoint downloads `deepseek-ai/DeepSeek-V4-Flash-0731` unless
`--no-download-model` is supplied. This is a very large checkpoint.

## What is measured

The Recomputed reference performs a full online prefill without treating
document 1 as a prefix. RedKnot materializes document 1 as a certified prefix and runs
all remaining documents (2–4 or 2–8, depending on context length) through the combined independent-position-0 head-reuse/online
RoPE relocation and merge path, Indexer-selected transformer rows, and
assignment-sparse adaptive expert Top-K. Dense boundary layers are 0–2 and
40–42; middle layers use eight online global heads and 56 reusable local heads.

The user-facing result compares every complete Recomputed output with the
matching complete RedKnot output side by side. It also reports
Recomputed/RedKnot TTFT,
speedup and the conservative arithmetic compute ledger used during the
original evaluation. The ledger gives no saving credit to Indexer,
compressor, router, normalization, memory traffic, kernel launch or TP
communication. Machine-readable low-level evidence remains in each
`result.json`; `summary.json` and `comparison.md` are the release reports.

A nonzero benchmark exit can mean a qualification gate failed even when
`result.json` was produced; the top-level matrix records both. Only a missing
`result.json` is treated as a fatal execution failure.

See `datasets/LongBench/PROVENANCE.json`,
`head_class/deepseek_v4_flash_0731_redknot.json`, and
`sparse_ffn_params/deepseek_v4_flash_0731.json` for the frozen contracts.
