# Frozen source assets

`sglang/` contains an exact-byte snapshot of selected benchmark assets from
[RedKnot-SGLang commit 19c152a44eee0439afc85fe95ac23ce5a914c8fd](https://github.com/rednote-machine-learning/RedKnot/tree/19c152a44eee0439afc85fe95ac23ce5a914c8fd/test/srt/redknot).
`sglang/SOURCE.json` records every original repository path, Git blob ID, byte
length, and SHA-256. Its canonical inventory digest is pinned in `catalog.py`.
Preparation verifies every file before it consumes any source selection.

The snapshot contains 152 original files (47,656,661 bytes):

- Four release suites: 64K, 128K, 256K and 440K, with their original SHA files
  and `RELEASE_SUITES.json`. Each suite has ten scored short-answer cases and
  five fixed-30-token, unscored showcases.
- The 38 referenced cohorts' `profile.json`, `data_selection.json` and
  `prompt_manifest.json` files. Old absolute paths in a profile are historical
  metadata; preparation resolves the immutable sibling assets in this folder.
- Five original LongBench JSONL files and their original `PROVENANCE.json`:
  HotpotQA, 2WikiMQA, MuSiQue, MultiFieldQA-en and TriviaQA. The packaged data
  files total 46,405,515 bytes. The four frozen release suites cover the first
  four datasets. TriviaQA is available in the explicitly new dataset matrix.
- The original head and sparse-FFN policy JSON files, and the source repository
  `LICENSE`. These policies are archival model-specific configurations; they
  are not validated or automatically applied to DeepSeek V4.1 in DeepInfer.

The original data provenance says the upstream LongBench download revision was
not recorded. This snapshot preserves that limitation and pins the bytes we
actually have; it does not invent a Hugging Face revision. The original
LongBench-derived RAG workload concatenates fixed document chunks and is not an
untouched official LongBench leaderboard prompt.

The release suites preserve their original curated case selection, including
each case's `selection_origin`. Replaying those cases does not make them a new
independently selected benchmark. The optional five-dataset matrix separately
selects the first usable physical JSONL rows in ascending order before any
inference. Its manifest records `output_blind_dataset_matrix`, and it is not
reported as an original frozen 15-case release.

## Migration protocol

`target-token-budget-protected-source-context-v1` regenerates a target workload
because the source DeepSeek V4 Flash 0731 tokenizer and prompt encoder differ
from the target DeepSeek V4.1 tokenizer and protocol. Original input hashes are
retained only as source provenance. Target input hashes are always computed
from newly generated target IDs.

For each original chunk, the migration retains the original physical row order
and reserves room for the **entire context of every source query row** in that
cohort. Non-query background contexts may be shortened to the new target-token
budget. If complete query contexts do not fit, preparation fails instead of
silently dropping their evidence. Additional background text comes from
distinct real rows in deterministic order after the source selection, wrapping
to remaining lower row IDs if needed. Each target row span records its original
row hash, offsets, completeness and selection origin.

The target chunks are decoded without removing special tokens, joined with the
original question instruction, rendered using the declared target chat
protocol, and encoded once as a complete request. Preparation additionally
checks that the original query context is present verbatim in the rendered
request. The document-token budget and the actual final input length are
reported separately: delimiters, instructions, chat framing and final
retokenization can change the latter. Runners must use the sealed final IDs
directly and may not tokenize them again.

Warmup uses real questions in reverse physical row order, excluding every query
row in the full original suite and any newly selected matrix queries. Warmup
questions and final input hashes are disjoint from evaluation. Background
documents can overlap; this is not a claim that the entire corpora are disjoint.
