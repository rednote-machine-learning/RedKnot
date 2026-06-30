# RedKnot on Qwen3.5-35B-A3B (hybrid MoE) — Sparse Attention + Sparse MoE

This documents how RedKnot is adapted to **Qwen3.5-35B-A3B**, a *hybrid* MoE model
(40 layers: 10 `full_attention` + 30 `linear_attention` GatedDeltaNet; MoE FFN with
512 experts, top-10, + a shared expert). Unlike dense Llama, the FFN is MoE and
3/4 of the attention layers are linear, so the standard RedKnot recipe had to be
re-derived. Three independent sparsity mechanisms are combined.

## TL;DR results (2-GPU bf16, multi-dataset)

| context | F1 (std→rk) | ΔF1 | full save | linear save | MoE save | **total FLOPs save** | theo-max TTFT |
|---|---|---|---|---|---|---|---|
| 16K (4×4K) | 0.750→0.821 | +0.07 | 11.7% | ~50% | ~8% | **~23%** | 1.31x |
| 32K (4×8K) | 0.875→0.875 | +0.00 | 17.9% | 52% | 9% | **~26%** | 1.36x |
| 64K (8×8K) | 0.750→0.750 | +0.00 | 20.9% | 57% | 11% | **~29%** | 1.40x |

- Accuracy is **lossless** at the shipped config (ΔF1 ≥ 0).
- `total FLOPs save` is **pure compute** (excludes cross-GPU comm / framework).
- `theo-max TTFT` = speedup if comm were free; measured wall-clock TTFT in this
  device_map (layer-sharded) multi-GPU setup is lower because comm dominates
  (~44% of wall time) and RedKnot only reduces compute, not comm.

---

## 1. Linear attention — per-head token window + decayed-prefix reuse

**Why it works.** GatedDeltaNet state `S_t = g_t·S_{t-1} + k_t⊗v_t` is a *fixed-size
matrix* that exponentially forgets old tokens at a per-(layer,head) rate. We
measured each head's decay → effective memory length `mem_len = 1/(1-decay)`.
The bottom ~60% of heads have `mem_len` of tens of tokens (fast-forget), the rest
are long-memory.

**Mechanism (per layer, head):**
- **GLOBAL head** (long memory): recompute full status (exact).
- **LOCAL head** (`mem_len < window`): the window-外 history is the **decayed
  prefix status** (a single matrix), reused as the window's initial state; only
  the window-内 tokens are recomputed. Validated numerically lossless
  (`verify_decayed_prefix_state.py`, rel-err 0).

**Why it's the biggest win.** Linear is ~42% of prefix FLOPs. A windowed head
saves `1 - window/T`; with `window ≪ context` this is 87–98% per head, so the
component saves ~52–57% overall — the dominant contributor.

**Implementation.** `install_linear_segmented` runs the native fla chunk kernel in
SEGMENTS with state relay (GLOBAL: accumulate; LOCAL: reset to decayed prefix at
window boundaries). Crucially the native kernel has a long-sequence perf cliff
(one T=20000 call ≈ 450× a sum of small segment calls), so segmenting is both
correct AND fast.

Static config: `sparse_ffn_params/qwen3.5-35B-A3B.json`
(`safety`, `min_window`, `dense_prefix_layers`, `segment`). Window per head =
`clamp(ceil(safety · mem_len), min_window)`; `≥ context` → GLOBAL.

## 2. Full attention — multi-head class (global / local), shallow-dense

**Why shallow-dense.** Early full layers carry critical features; sparsifying them
corrupts the stream. Deep full layers tolerate head-class sparsity.

**Mechanism.** First `dense_full_layers` full layers are exact; the remaining
(deep) full layers use head classification:
- **global head** (`frac_global` of kv-heads): attends the whole prefix.
- **local head**: sink + sliding `window`.

Sweet spot: keep the **first half + 1** full layers dense (6 of 10), sparsify the
deep 4 with `frac_global=0.4, window=4096`. Lossless. Full attention is only ~6%
of FLOPs at 32K (its O(L²) only dominates at much longer context), so its absolute
contribution is small but free.

Static config: `head_class/qwen3.5-35B-A3B_redknot.json`
(`dense_full_layers`, `frac_global`, `window`).

## 3. MoE — deep token sparsity via attention-mass importance

**The hard part: a token-importance criterion.** router top-k mass and token norm
have NO spread on Qwen3.5 (512-expert routing is flat; RMSNorm flattens norm). The
signal that DOES have spread is **attention mass** (how much a key token is attended
to): max ≈ 1400× mean, top-1% of tokens hold ~17% of attention. So a token's
importance is read from the **deep full-attention layers' mass**.

**Mechanism (two-pass).**
1. Pass-1: collect per-token attention mass averaged over the deep half of full
   layers → global token-importance (`collect_attention_mass`).
2. Pass-2: deep MoE layers (`deep_moe_start_layer`+) let **low-mass tokens skip
   the routed experts** (keep only the cheap shared expert); high-mass tokens run
   the full MoE (`install_moe_token_sparse`).

MoE is ~35% of FLOPs (largest block), but only the deep half is sparsified and
only ~20% of tokens skip, so the saving is modest (~9–11%) and is the most
accuracy-sensitive knob (multi-hop questions degrade first). `mass_thresh`
trades accuracy for saving.

---

## Compute/accuracy trade-off

Sweeping the three knobs (`sweep_compute_accuracy.py`):

| strength | ΔF1 | total FLOPs save |
|---|---|---|
| lossless (shipped) | ~0 | ~23–29% (grows with context) |
| mild loss (ΔF1≈-0.05) | -0.05 | ~19% (older formula) → higher with corrected linear |
| aggressive | -0.10 | bounded by proj/norm (17%, untouched) and MoE accuracy |

50% total saving is NOT reachable losslessly on this architecture: proj/norm
(~17%) is untouched, MoE can't be aggressively token-sparsified without accuracy
loss, and linear is already near its limit. Realistic lossless ceiling ≈ 25–30%.

---

## One-click run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
REDKNOT_N_SAMPLES=1 \
REDKNOT_MAX_NEW=8 \
REDKNOT_COMPILE=0 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/workspace/redknot/redknot-0.1/python:/workspace/redknot/redknot-0.1/.venv_tf5/lib/python3.11/site-packages:/root/miniconda3/lib/python3.11/site-packages \
/workspace/redknot/redknot-0.1/.venv_tf5/bin/python \
  /workspace/redknot/redknot-0.1/test/srt/redknot/benchmark_RedKnot_QWen35_RAG.py
```

Outputs per config: input length, dataset, baseline/RedKnot text outputs,
compute saving, and TTFT speedup.

## Files

| file | role |
|---|---|
| `benchmark_RedKnot_QWen35_RAG.py` | one-click benchmark (3 configs, multi-dataset) |
| `run_qwen35_rag.sh` | launcher (tf5 venv + 2 GPU) |
| `head_class/qwen3.5-35B-A3B_redknot.json` | full-attention sparsity config |
| `sparse_ffn_params/qwen3.5-35B-A3B.json` | linear + MoE sparsity config |
| `python/.../redknot/driver_qwen35.py` | all RedKnot-Qwen3.5 mechanisms |

## Environment note

Qwen3.5 (`qwen3_5_moe`) needs transformers ≥5 (not 4.57). An isolated venv
`.venv_tf5` (transformers 5.10.2 + system torch) is used; the main env stays on
4.57.1 so Llama/Qwen3 benchmarks are unaffected. `HF_HUB_OFFLINE=1` avoids a slow
transformers-5 import-time directory scan on the networked FS.
