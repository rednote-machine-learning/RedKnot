# DeepSeek-V4-Pro-0813 frozen qualification profiles

These are the two canonical single-corpus inputs for the Pro 440K and 512K
qualification gates.  `redknot_multidataset_profile_v2` is the historical
profile format name; one profile still represents one dataset, one reusable
offline prefix, and one or more queries.  The mixed 15-case release suite is a
separate ordered runner product and must not be passed as
`REDKNOT_QUALIFICATION_PROFILE`.

Both profiles are output-blind HotpotQA cohorts with ten fully-contained,
unique query rows.  They are intended for `full_combined_production_v1`.
`zoff_only` remains an explicitly diagnostic execution profile and cannot
support the formal Pro result.

| Target | Frozen profile | Query rows | Profile SHA-256 |
| --- | --- | --- | --- |
| 440K (`450560`) | `pro0813_440k_hotpotqa_10q/profile.json` | `0,4,9,14,19,24,30,34,1,5` | `52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b` |
| 512K (`524288`) | `pro0813_512k_hotpotqa_10q/profile.json` | `0,5,11,16,22,29,34,39,1,6` | `e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d` |

The checked-in `profile.json.sha256` files bind the exact profile bytes.  Each
profile in turn binds the byte SHA of its co-located `data_selection.json` and
`prompt_manifest.json`, their canonical semantic digests, the raw LongBench
dataset SHA, exact query quota/row IDs/cases, and these files from the pinned
official Pro revision `72e1d3230f6c080a530b0a1d46f8eb4602340597`:

- `encoding/encoding_dsv4.py`:
  `abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`
- `tokenizer.json`:
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
- `tokenizer_config.json`:
  `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`
- tokenizer runtime: `tokenizers==0.22.1`

The official Pro tokenizer files are byte-identical to the Flash tokenizer
files previously used to freeze the 440K release cohort.  Rebuilding the 440K
data selection therefore reproduces its old data manifest byte-for-byte and
all ten prompt cases/token hashes exactly; only the previously embedded Flash
filesystem paths and resulting prompt-manifest digest change to the canonical
Pro path.  The same raw HotpotQA file contains enough fully-contained material
for the independent 512K cohort: the builder produced eight complete 65,536
token chunks and ten unique queries without padding, row reuse, or fabricated
records.

## Fail-closed CPU preflight

Run this before any holder release or GPU action:

```bash
python test/srt/redknot/verify_pro0813_qualification_profile.py \
  /workspace/RedKnot_Pro0813/test/srt/redknot/qualification_profiles/pro0813_440k_hotpotqa_10q/profile.json \
  --expected-target-tokens 450560 \
  --expected-profile-sha256 52417a8af4a26d3ea109d1993fb88b9acdc60f7b82f651adbd21ff5a483e9a7b

python test/srt/redknot/verify_pro0813_qualification_profile.py \
  /workspace/RedKnot_Pro0813/test/srt/redknot/qualification_profiles/pro0813_512k_hotpotqa_10q/profile.json \
  --expected-target-tokens 524288 \
  --expected-profile-sha256 e8876e12107f05ceec36f1e758002430221d6a490e431a6fc8662cbde5c0703d
```

The verifier rejects a relative profile argument, symlinks, directory escape,
duplicate JSON keys, target/geometry drift, profile or artifact mutation,
dataset quota/sample drift, raw-row mutation, prompt/data digest mismatch,
tokenizer identity drift, and a non-pinned tokenizer runtime.

## Deterministic regeneration

Regeneration is CPU-only.  It refuses to replace a different frozen file.  To
audit a rebuild, target an empty temporary directory and compare all four
outputs byte-for-byte with the checked-in directory.

```bash
python test/srt/redknot/prepare_multidataset_512k_manifests.py \
  --core test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py \
  --dataset hotpotqa \
  --data-dir test/srt/redknot/datasets/LongBench/data \
  --model /workspace/Models/DeepSeek-V4-Pro-0813 \
  --num-queries 10 --row-offset 0 --cohort-index 44000 \
  --chunk-tokens 56320 --num-chunks 8 \
  --data-out /tmp/pro0813-440k/data_selection.json \
  --prompt-out /tmp/pro0813-440k/prompt_manifest.json \
  --profile-out /tmp/pro0813-440k/profile.json

python test/srt/redknot/prepare_multidataset_512k_manifests.py \
  --core test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Pro0813_RAG.py \
  --dataset hotpotqa \
  --data-dir test/srt/redknot/datasets/LongBench/data \
  --model /workspace/Models/DeepSeek-V4-Pro-0813 \
  --num-queries 10 --row-offset 0 --cohort-index 51200 \
  --chunk-tokens 65536 --num-chunks 8 \
  --data-out /tmp/pro0813-512k/data_selection.json \
  --prompt-out /tmp/pro0813-512k/prompt_manifest.json \
  --profile-out /tmp/pro0813-512k/profile.json
```

`test_pro0813_qualification_profiles.py` performs the byte-for-byte rebuild,
both positive closures, consumer path resolution, and negative tamper/escape
checks without touching CUDA.
