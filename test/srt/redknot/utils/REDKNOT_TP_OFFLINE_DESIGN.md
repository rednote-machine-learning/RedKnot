# RedKnot TP-Native Offline-Segment Sparse Prefill — Implementation Design

Status: **Phase 1 DONE (request plumbing). Phase 2 designed below (NOT yet implemented).**

This document specifies how to finish wiring RedKnot's *true* sparse attention
path so it works under sglang tensor parallelism (TP), driven through the
public `Engine` API. It is the continuation of the request-plumbing work that
is already merged (Phase 1).

---

## 0. Background / why this is needed

The `redknot` attention backend already implements head-classified sparse
prefill and is TP-aware. Its sparse path only activates when a request carries
**offline segment ids** via `forward_batch.redknot_offline_segments`. Without
them, `RedKnotAttnBackend.forward_extend` falls back to per-request dense SDPA
(`_sdpa_single_extend`), which is *slower* than the fused baseline and yields
no sparsity benefit. This is exactly what was observed on Qwen3.5 (32K/64K:
TTFT 0.89–0.92x, i.e. a slowdown) — the "redknot" runs were silently dense.

To get the real benefit (and a meaningful TTFT comparison) the offline document
chunks must be pre-filled once, their per-layer KV captured, registered in the
`OfflineKVCache`, and then referenced per query.

Under TP the *only* viable way to build that KV on a model that does not fit on
one GPU (e.g. 397B) is to capture it from sglang's own sharded prefill — there
is no HuggingFace model in the scheduler process.

---

## 1. What Phase 1 already delivered (request plumbing)

End-to-end carriage of `redknot_offline_segments` from the Engine API down to
the forward pass. Files touched:

| File | Change |
|---|---|
| `python/sglang/srt/entrypoints/engine.py` | `generate()` gains `redknot_offline_segments=` param; threaded into `GenerateReqInput`. |
| `python/sglang/srt/managers/io_struct.py` | New field on `GenerateReqInput` + `TokenizedGenerateReqInput`; `_normalize_redknot_offline_segments()`; `__getitem__` slicing. |
| `python/sglang/srt/managers/tokenizer_manager.py` | Copies field into `TokenizedGenerateReqInput`. |
| `python/sglang/srt/managers/schedule_batch.py` | `Req.__init__` stores `self.redknot_offline_segments`. |
| `python/sglang/srt/managers/scheduler.py` | Threads field from `recv_req` into `Req`. |
| `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardBatch.init_new` builds `redknot_offline_segments` from `batch.reqs` (mirrors `lora_ids`). |

**Consumer side (already present, unchanged):**
`RedKnotAttnBackend.init_forward_metadata` reads
`forward_batch.redknot_offline_segments` and `_redknot_single_extend` splices
the offline KV. RoPE realignment helper is auto-bound in
`model_runner.py:2261` (`attach_rope_helper`).

**Net effect after Phase 1:** a request *can* carry segment ids, and the
backend *will* consume them — but the segments themselves are never built or
registered, so nothing is in the `OfflineKVCache` to reference yet. Phase 2
fills that gap.

---

## 2. The data contract the consumer expects

From `redknot_backend.py::_redknot_single_extend` (lines ~652–710):

- `seg = self.offline_cache.to_device(sid, device)` — looked up by string id.
- `k_off, v_off = seg.kv[layer.layer_id]` — **one (K,V) pair per model layer**,
  indexed by *global* `layer.layer_id`.
- Each `k_off`/`v_off` shape: **`[1, KVH_per_rank, L, head_dim]`** where
  `KVH_per_rank == layer.tp_k_head_num` (i.e. already TP-sharded heads).
- KV must be stored as if RoPE were applied at absolute positions `[0, L)`.
  The consumer calls `_rope_helper.reposition_offset(k_off, src_start=0,
  dst_start=cursor, length=seg.doc_len)` to move it to its runtime position.
- `seg.doc_len == L` (number of tokens in the segment).
- `OfflineSegment` is built via
  `offline_cache.build_offline_segment(segment_id, token_ids, kv)` and stored
  with `OfflineKVCache.put(seg)`. The cache is a **per-process singleton**
  (`get_global_offline_cache()`), so under TP each rank stores only its own
  head shard under the *same* `segment_id`.

Segment id:
`OfflineKVCache.compute_segment_id(model_id, token_ids, prepend_bos=...)`
— deterministic hash; **must be identical across ranks** (it is, since it only
depends on model_id + token ids, not on shard).

---

## 3. Phase 2 design — building offline segments natively under TP

### 3.1 Strategy

Add an **offline-prefill request mode** to the scheduler: prefill a document
chunk through the normal sglang forward path, then snapshot its per-layer KV
out of `token_to_kv_pool` (which is already this rank's head shard), build an
`OfflineSegment`, register it, and free the request's KV slots.

This reuses the entire existing prefill stack (TP all-reduce, MoE, GDN linear
layers, FP8) — so numerics match a real prefill by construction.

### 3.2 New control message: `BuildOfflineSegmentReqInput`

`io_struct.py`:

```python
@dataclass
class BuildOfflineSegmentReqInput(BaseReq):
    # The document chunk to prefill, as text or token ids.
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    prepend_bos: bool = False
    model_id: Optional[str] = None   # defaults to served model path
    rid: Optional[str] = None

@dataclass
class BuildOfflineSegmentReqOutput:
    segment_id: str
    doc_len: int
```

Engine method:

```python
def build_offline_segment(self, text=None, input_ids=None, prepend_bos=False):
    obj = BuildOfflineSegmentReqInput(text=text, input_ids=input_ids,
                                      prepend_bos=prepend_bos)
    return self.loop.run_until_complete(
        self.tokenizer_manager.build_offline_segment_request(obj))
```

The tokenizer_manager tokenizes (if `text`) and forwards to the scheduler; the
scheduler runs the offline prefill on every TP rank and returns the
`segment_id` (computed identically on each rank; rank 0 replies to the client).

### 3.3 Scheduler: offline prefill + KV snapshot

Add `Scheduler.handle_build_offline_segment(recv_req)`:

1. Build a one-off `Req` with `origin_input_ids = chunk_ids`, a sampling spec of
   `max_new_tokens=0` (prefill only). Mark it `is_offline_build=True`.
2. Run a single extend/prefill forward for just this req (can reuse
   `prepare_for_extend` + `run_batch` with a 1-req batch, or a dedicated
   minimal path). Do **not** sample or stream.
3. Snapshot KV from the pool. For each global layer `li`:
   ```python
   tok_locs = req_to_token[req_pool_idx, :L]            # this req's slots
   k_pool = token_to_kv_pool.get_key_buffer(li)         # [N, KVH_rank, D]
   v_pool = token_to_kv_pool.get_value_buffer(li)
   k = k_pool[tok_locs].movedim(1, 0).unsqueeze(0)      # -> [1, KVH_rank, L, D]
   v = v_pool[tok_locs].movedim(1, 0).unsqueeze(0)
   kv.append((k.contiguous().clone(), v.contiguous().clone()))
   ```
   Note: clone, because the slots will be freed.
4. `sid = OfflineKVCache.compute_segment_id(model_id, chunk_ids,
   prepend_bos=...)` (same on all ranks).
5. `seg = build_offline_segment(segment_id=sid, token_ids=chunk_ids_tensor,
   kv=kv)`; `get_global_offline_cache().put(seg)`.
6. Release the req's KV slots (`token_to_kv_pool.free(...)`,
   `req_to_token_pool.free(...)`).
7. Rank 0 returns `BuildOfflineSegmentReqOutput(sid, L)`.

### 3.4 The RoPE subtlety (the riskiest part — verify carefully)

The consumer assumes offline KV is rotated for positions `[0, L)`. A normal
sglang prefill of a standalone chunk (with `extend_prefix_lens=0`) *does* rotate
the chunk at positions `[0, L)` — so snapshotting after a fresh prefill of the
chunk alone already satisfies the `[0, L)` contract. **This must be verified**:
confirm the offline-build req is prefilled with prefix_len=0 and positions
starting at 0 (no radix-cache prefix reuse — disable radix cache for these
requests or ensure a clean slot).

If positions are not `[0, L)`, either:
- (a) force position offset 0 for offline-build reqs, or
- (b) unrotate at snapshot time using the bound `RoPEHelper`
  (`reposition_offset(k, src_start=pos0, dst_start=0, length=L)`).

Option (a) is cleaner and preferred.

### 3.5 FP8 KV note

If `kv_cache_dtype` is fp8, `get_key_buffer` returns fp8. The consumer kernels
(`_kernel_fn`, FA2/FA3) expect a consistent dtype; the offline segment KV must
match what online KV uses. Since both come from the same pool, dtypes already
match — store fp8 as-is. Verify the segment-splice `torch.cat` (backend lines
676–677) does not upcast unexpectedly.

---

## 4. Phase 3 — validation plan (do NOT skip)

Numeric correctness must be proven before trusting any TTFT/accuracy number.

1. **Single-layer parity (35B, tp=2):**
   - Pick one RAG sample, 2 doc chunks.
   - Path A: normal full prefill of `concat(docs)+query` → capture last-token
     logits.
   - Path B: `build_offline_segment` for each doc → `generate(query,
     redknot_offline_segments=[sids])` with a head config that marks **all
     heads dense/global** (so RedKnot is mathematically equivalent to full
     attention).
   - Assert top-1 token matches and logit max-abs-diff < tol (e.g. 1e-2 for
     bf16). This isolates the KV-snapshot + splice correctness from sparsity.

2. **Sparse correctness (35B, tp=2):** switch to the real head config
   (local/global mix). Compare F1 on a few samples vs the dense baseline; expect
   near-parity (RedKnot is designed to be near-lossless).

3. **TTFT measurement:** with offline segments pre-built (offline time
   excluded), measure online prefill TTFT for baseline vs redknot at 16K / 32K /
   64K / 128K. Expect redknot TTFT speedup to grow with context length.

4. **Scale to 397B-FP8 (tp=8):** rerun (1)+(3) at 64K/128K.

Verification harness location: extend
`test/srt/redknot/benchmark_RedKnot_Qwen35_sglang_TP.py` with an
`offline-segment` mode (build segments first, then query with segment ids).

---

## 5. File-by-file work checklist for Phase 2/3

- [ ] `io_struct.py`: `BuildOfflineSegmentReqInput` / `...Output`.
- [ ] `entrypoints/engine.py`: `Engine.build_offline_segment()`.
- [ ] `managers/tokenizer_manager.py`: `build_offline_segment_request()` +
      route the new message; tokenize text → ids.
- [ ] `managers/scheduler.py`: recv-loop dispatch +
      `handle_build_offline_segment()` (prefill, snapshot, register, free).
- [ ] `mem_cache` access: confirm `token_to_kv_pool.get_key_buffer/
      get_value_buffer(layer_id)` and `req_to_token_pool.req_to_token` are
      reachable from the scheduler/worker for snapshotting per rank.
- [ ] Ensure offline-build reqs bypass radix prefix reuse (clean positions).
- [ ] `benchmark_RedKnot_Qwen35_sglang_TP.py`: add `offline-segment` method
      (build then query) + TTFT at multiple context lengths.

---

## 6. Risk register (ranked)

| Risk | Severity | Mitigation |
|---|---|---|
| RoPE position of snapshotted KV ≠ `[0,L)` | HIGH | Force prefix_len=0 for offline-build reqs; verify with parity test (§4.1). |
| Paged KV snapshot indexing wrong per rank | HIGH | All-dense head config parity test isolates this. |
| FP8 KV dtype mismatch on splice | MED | Both KVs from same pool; assert dtype equal. |
| Radix cache reuses prefix → wrong positions | MED | Disable radix cache for offline-build reqs. |
| Cross-rank segment_id divergence | LOW | id depends only on model_id+token ids (shard-independent). |

---

## 7. Effort estimate

Phase 2 (build mode + snapshot): ~5–7 focused days, dominated by the RoPE /
paged-KV parity debugging in §4.1. Phase 3 validation: ~2–3 days. This is a
new-subsystem-level change requiring per-layer numeric verification on a real
model; it is intentionally NOT attempted as a single-shot edit.
