# Full and RedKnot S1.6 on LongBench QA

Replay the 300 measured RAG prompts with DeepSeek-V4.1-Flash-1017. The default run loads the questions from the bundled archive, prepares the document KV banks, and generates a Full and S1.6 answer for every question. It uses the original 75 four-request batches, eight expert-parallel ranks, D-Spark with five speculative tokens, and the main and draft CUDA graphs. Both methods share a resident model and see identical token sequences.

## Run

On an eight-GPU H200 host, activate the DeepInfer environment and point the entry at your local model. First-time setup and native build instructions are in the [environment guide](ENVIRONMENT.md).

```bash
python test/benchmark_rag_redknot.py --model-path /models/DeepSeek-V4.1-Flash-1017
```

The command prints Full and S1.6 scores, mean TTFT and speedup by dataset. Open the printed `report/index.html` and click a dataset to compare questions, references and both full answers side by side. `report/answers/*.jsonl` also preserves raw decoded text, output token IDs, first divergent token, output length, stop reason, F1, TTFT, total duration and decode rate for each pair. `report/summary.json` groups results by dataset, length and prompt construction. `results/groups/` holds fresh generation records. Each invocation creates a new result directory under `benchmark-output/redknot-s16-300/`. To choose a destination, pass `--output /path/to/new-run`.

To reproduce the scoring and inspect the 600 recorded answers on a CPU, run the reference command. The report labels these timings as historical GPU measurements. To audit the packaged inputs and B4 plan, run the check command. Both modes work offline with Python 3.12 and its standard library.

```bash
python test/benchmark_rag_redknot.py --reference
python test/benchmark_rag_redknot.py --check
```

## Frozen workload

HotpotQA, 2WikiMultihopQA and MuSiQue contribute 100 unique source questions each. Each request contains 5–15 natural chunks. Of the 300 requests, 28 retain the original LongBench prompt and 272 add complete contexts from the same subset. The report splits the two populations. Actual inputs span **12,776–130,269 tokens**. The configured 16K, 32K, 64K and 128K bands contain 56, 64, 60 and 120 requests respectively. Four of the 75 original batches span a dataset boundary.

`redknot_s16_longbench_300.zip` contains the source texts, templates, questions, reference answers, exact input token IDs, chunk boundaries and B4 order. Its SHA-256 is `ea528890ab935dbc6a225ab4a7c1cf54b8654cc6d74f0cb79982eabe7401b0d0`. The source texts and templates reconstruct all 22,167,716 input tokens. Dataset-specific licensing, attribution, upstream revisions and full license texts are in the archive’s `NOTICE.txt`, `provenance.json` and `licenses/`. HotpotQA uses CC BY-SA 4.0, MuSiQue uses CC BY 4.0, and 2Wiki uses Apache-2.0. The LongBench template notice is MIT. [LongBench](https://github.com/THUDM/LongBench) · [HotpotQA](https://hotpotqa.github.io/) · [2Wiki](https://huggingface.co/datasets/xanhho/2WikiMultihopQA) · [MuSiQue](https://github.com/StonyBrookNLP/musique)

## Recorded eight-GPU results

| Dataset | Questions | Full F1 | S1.6 F1 | Full mean TTFT | S1.6 mean TTFT | TTFT speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 100 | 75.61 | 72.23 | 31.623 s | 14.412 s | 2.194× |
| 2WikiMultihopQA | 100 | 66.22 | 55.92 | 32.960 s | 16.028 s | 2.056× |
| MuSiQue | 100 | 46.05 | 48.66 | 32.695 s | 15.827 s | 2.066× |
| All | 300 | 62.63 | 58.94 | 32.426 s | 15.423 s | 2.102× |

Scores are LongBench v1 English QA token F1 on a 0–100 scale, maximized over the references for each question. Speedup divides Full mean TTFT by the S1.6 mean for the same questions. TTFT is measured from request receipt to the first raw output token with the model resident and S1.6 document KV prepared separately. The runner also records offline preparation, per-request total duration and decode rate. Compare new measurements under the same hardware, software and cache conditions.
