# Head-paged KV ownership and sharing

This package implements an experimental bounded MHA/GQA KV manager and a
direct paged attention backend. Pages belong to physical KV heads/groups,
not to query heads. The existing RedKnot `segpaged_attention` entry point
dispatches to this backend when given `ManagedSegPagedKVCache`. The regular
`RedKnotAttnBackend` and `SegPagedAttnBackend` also expose an explicit shared
request path for both prefill/extend and decode.

## Ownership and mutation

* A request owns an immutable version of its segment table. A segment is
  addressed by `(layer, kv_head, occurrence)` and records target logical
  positions (not compacted storage offsets) and adapter-supplied provenance.
* Forks and arbitrary segment occurrences can reference the same sealed
  physical pages. Sharing is not restricted to prefixes. A partial repair
  reserves only touched pages; a full-page rewrite copies no old rows.
  Multi-page repairs batch row gathering/scattering across all touched
  pages, avoiding one set of GPU launches per page. Temporary payloads
  contain only touched retained rows and compact repair rows.
* Every published page, including a partial tail, is immutable. Append
  detaches a partial tail and retains full historical pages. Repeated
  occurrences in one root have one physical owner reference per page.
* Multi-head updates validate and reserve their entire destination set
  before copying. An OOM does not partially change the request root.
  A write transaction publishes only after its completion event and only
  if the request generation and version still match.
* Read leases pin a complete version until the consumer's CUDA event.
  Owner references, reader pins and pending device events are separate.
  Event failures quarantine the pool; expiry of a network lease does not
  imply that a GPU page can be overwritten.

The pool allocates fixed K/V slabs `[capacity_pages, page_size, head_dim]`.
`allocated_bytes` counts active page capacity; `reserved_slab_bytes` is the
actual fixed backing allocation. Forking saves page admission capacity
without shrinking an already allocated CUDA slab.

```python
import torch
from sglang.srt.mem_cache.head_kv import HeadKVManager, HeadPagePool
from sglang.srt.mem_cache.head_kv.manager import SegmentWrite, SegmentPatch

pool = HeadPagePool(128, 16, 128, dtype=torch.bfloat16, device="cuda")
manager = HeadKVManager(pool)
manager.create_request("parent", context_id="context-hash",
                       contract="my-model-and-layout-v1", namespace="tenant-a")
k = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
key = (0, 0, "document-occurrence-0")
manager.update("parent", writes=[SegmentWrite(
    key, k, v, tuple(range(32)), "adapter-state-hash")])
manager.fork("parent", "child")
manager.update("child", patches=[SegmentPatch(
    key, (19,), k[:1], v[:1], "repaired-state-hash")])
manager.release_request("child")
manager.release_request("parent")
pool.collect()
```

Adapters must pass their model's actual namespace and compatibility contract.
The strings in this example are placeholders, not computed validity proofs.

## Attention

`ReadLease.descriptor(layer, num_kv_heads)` provides page slots, extents and
logical positions. `paged_attention` reads the slabs directly, with GQA
mapping, per-head windows/sinks, causal masking and online softmax across
pages. It does not concatenate the historical KV sequence. `gather` exists
only as an explicit reference/export helper.

The Triton kernel accumulates QK, softmax and PV in FP32 before casting the
output. Different low-precision attention backends can round intermediate
values differently; model-level comparisons must report both logit error
and token agreement. The public API validates descriptors (including a CPU
sync on CUDA). This initial implementation targets correctness, decode and
small query batches, not optimized dense prefill or production throughput.

## RedKnot and SegPaged backend integration

The registered `redknot` and `segpaged` backends accept a caller-owned manager.
Set `runner.redknot_shared_kv_manager` **before backend construction**; the
registry passes it into `RedKnotAttnBackend` or `SegPagedAttnBackend`. Direct
construction accepts `shared_kv_manager=manager` as well. This is an opt-in
integration API, not a new command-line switch or automatic scheduler setup.

```python
import torch
from sglang.srt.layers.attention.attention_registry import create_redknot_backend
from sglang.srt.mem_cache.head_kv import HeadKVManager, HeadPagePool

# runner is an existing, configured ModelRunner. Choose capacity for the
# physical KV heads of all participating local layers, with room for COW.
pool = HeadPagePool(4096, 16, 128, dtype=torch.float32, device=runner.device)
runner.redknot_shared_kv_manager = HeadKVManager(pool)
backend = create_redknot_backend(runner)
# create_segpaged_backend(runner) selects the same ownership adapter.
shared = backend.shared_kv
```

The pool dtype/device and equal Q/K/V head dimension must match the actual
model tensors. The FP32 example reflects the currently qualified model
comparison; BF16 model qualification remains incomplete as described below.
The contract must identify weights, adapters, KV representation and position
semantics. The namespace is the isolation domain, and the context ID identifies
the already-computed context. These values are supplied by the integration;
the backend does not derive them from a token-pool slot.

The following illustrates the lifecycle at the attention boundary. `layer`,
projected Q/K/V, and prefill/decode batches come from the model's existing
forward loop. Apply the same request handles to every participating model
layer, and complete the parent's model prefill before forking it.

```python
parent = shared.create_request(
    "request-parent", context_id="canonical-context-identity",
    contract="weights-adapters-kv-position-contract", namespace="tenant-a",
)
handles_to_release = [parent]
try:
    prefill_batch.redknot_shared_kv_handles = [parent]
    # Run for each layer as its Q/K/V become available. They already include
    # the model's position transform. Batch metadata supplies explicit positions.
    backend.init_forward_metadata(prefill_batch)
    prefill_output = backend.forward_extend(
        q, k, v, layer, prefill_batch, save_kv_cache=True,
    )

    # After all parent layers finish: both children share its immutable pages.
    left = shared.fork_request(parent, "request-left")
    handles_to_release.append(left)
    right = shared.fork_request(parent, "request-right")
    handles_to_release.append(right)
    decode_batch.redknot_shared_kv_handles = [left, right]
    # Q/K/V contain one new token for each request in this same order.
    backend.init_forward_metadata(decode_batch)
    decode_output = backend.forward_decode(
        q_decode, k_decode, v_decode, layer, decode_batch, save_kv_cache=True,
    )
    # Later appends or shared.repair_request(handle, patches) detach only
    # changed pages. Repairs require model-correct upstream inputs.
finally:
    # The scheduler invokes release for completion, cancellation and errors.
    for handle in reversed(handles_to_release):
        shared.release_request(handle)
    pool.collect()  # In-flight reader events can defer physical reclamation.
```

A `SharedKVRequestHandle` contains the stable request ID and its generation.
Recreating an ID cannot make an old handle valid. A fork preserves the parent's
context/contract/namespace; it does not authorize reuse for an unrelated prompt.
A batch must carry exactly one distinct live handle per request, in token order.
Extend uses `positions`, `extend_seq_lens`, `extend_prefix_lens` and `seq_lens`;
decode uses one token per handle, `positions` and `seq_lens`. Every layer/head
must already cover its complete declared prefix. The adapter neither imports
an absent dense-cache prefix nor silently creates a request.

The registered backend dispatches this path before dense KV reads/writes.
It stores one persistent `__redknot_live__` segment per physical KV head/layer,
atomically appends that request's heads with `append_many`, and reads them via
`ManagedSegPagedKVCache` page descriptors. It never expands GQA KV into query
heads or gathers full historical KV. Local/global/dense head policies and
model scaling are preserved. The model's sliding window further restricts a
head window; sink policies that would expand a model sliding window are rejected.
Retrieval policies, legacy dense `redknot_offline_segments` splice plans and
extra model attention keyword arguments with non-`None` values are rejected.

This adapter requires eager execution and explicit lifecycle wiring. It rejects
unsupported cross-attention, encoder/bidirectional attention, speculative trees,
multiaxis positions, quantized/scaled KV, logit capping, unequal Q/K/V dimensions
and graph capture/replay paths. Supplying handles to an unconfigured backend also
fails instead of falling through to dense execution.

An append is atomic across one request's physical heads in one layer. Entire
batches and model forwards are **not** transactions: a later OOM or kernel error
can occur after earlier requests/layers committed. The caller must abort affected
requests or restore a checkpoint before retrying; validation failures detected
by the adapter's whole-batch preflight occur before its first append.

The existing scheduler still needs to allocate/admit requests, attach the right
handles, fork only valid contexts, retire requests and coordinate other model
state. It can still allocate its original dense KV slab even when these attention
calls bypass it. Therefore pool admission savings are not evidence of an
end-to-end serving memory reduction until that allocation and lifecycle are
integrated and measured.

## Cross-process sharing

`distributed.py` implements an HTTP owner and durable SQLite authority;
`transfer.py` connects it to real page tensors:

1. `export_snapshot` pins a request, stages page bytes on the host and puts
   content-addressed immutable objects at the owner. It publishes a manifest
   through a durable PREPARED/COMMITTED decision.
2. `import_snapshot` acquires and renews a retention grant, checks the
   manifest and compatibility contract, reserves all destination pages,
   verifies payload digests/layouts and copies into the local pool.
3. Only a fully received snapshot whose device copies completed becomes
   visible. Cancellation, malformed payloads and OOM leave the destination
   unpublished. Imported pages then use the normal local COW protocol.
4. Grants and active transfers prevent owner-side collection. Publication
   intents and request migration decisions are durable and idempotent.
   Prepared intents require explicit resolution; their expiry cannot
   silently discard an unresolved decision.

```python
from sglang.srt.mem_cache.head_kv.distributed import (
    KVShareStore, KVShareServer, KVShareClient,
)
from sglang.srt.mem_cache.head_kv.transfer import export_snapshot, import_snapshot

store = KVShareStore("/path/to/durable-owner", "tenant-a", capacity_bytes=1 << 30)
with KVShareServer(store) as owner:
    client = KVShareClient(owner.url, "tenant-a")
    manifest = export_snapshot(source_manager, "source", client,
                               operation_id="unique-publication-id")
    # Destination must be empty and have matching namespace/context/contract.
    import_snapshot(destination_manager, "destination", client, manifest,
                    holder="destination-worker")
store.close()
```

Use a new operation ID for a new publication. Retries of one operation keep
the same ID. Non-loopback listeners require an authentication token; the
transport is plaintext HTTP and needs a trusted network or an authenticated
TLS tunnel/proxy. The token does not attest the mathematical validity of KV.

RPC/metadata/object/in-flight budgets are enforced. A snapshot currently
supports up to 4096 unique objects, 16384 descriptor page references and
4 million logical positions; metadata is limited to 768 KiB within a 1 MiB
RPC. Logical positions use arithmetic runs. This is a bounded initial
protocol, not an unlimited long-context checkpoint format.

The authority also provides prepare/ready/commit-or-abort migration,
epoch fencing and exactly-once output-frontier advancement. Readiness is
an adapter attestation: the serving adapter must ensure all required GPU
and parallel ranks are ready before acknowledging it. A timeout does not
authorize a second request owner. The authority is one process with a
durable local database, not a replicated consensus service.

## Validity and integration boundaries

Sharing and COW preserve bytes and isolation; they do not establish that
those bytes are correct for a new prompt. Exact reuse requires the same
declared context and positions. Cross-context reuse requires an explicit
trusted adapter certificate/policy, and approximation lineage must remain
visible after further copies. Provenance/certificates are trusted adapter
attestations, not cryptographic or automatically derived model proofs.

The model adapter still owns cross-layer invalidation, repair inputs, head
policy selection, position transforms and quality validation. Relocating
position-encoded keys without a materialized adapter transform is rejected.
This package does not treat MLA latent state or recurrent/native bundles
as independent query-head KV pages.

This change provides managed SegPaged attention and opt-in paths in both
registered RedKnot/SegPaged backends. It does not automatically replace the
SGLang scheduler's existing request/KV pools. It does
not yet implement TP/CP/PP serving integration, scheduler-wide atomic
model steps, RDMA/GPU-direct transfer, authority replication/failover,
quantized-page mutation, or an MLA-specific backend. The Qwen3 validation
uses an explicit model-instance adapter to exercise real weights.

## Validation commands

From the repository root, with PyTorch and pytest installed:

```bash
python -m pytest -q test/srt/redknot/test_head_kv_manager.py \
  test/srt/redknot/test_head_kv_attention.py \
  test/srt/redknot/test_head_kv_distributed.py \
  test/srt/redknot/test_head_kv_transfer.py \
  test/srt/redknot/test_shared_kv_adapter.py \
  test/srt/redknot/test_shared_kv_backend.py
python test/srt/redknot/benchmark_head_kv_manager.py \
  --device cuda:0 --dtype bfloat16 --output benchmark.json
python test/srt/redknot/validate_head_kv_qwen3.py \
  --model-path /path/to/Qwen3-8B --device cuda --dtype bfloat16 \
  --output qwen3.json
```

Tests cover local ownership/COW/events, direct attention, real HTTP
transfers and faults, restart decisions, migration fencing and snapshot
validation. The shared-backend suites exercise registered entry-point dispatch,
chunked prefill, batched fork/decode, generation fencing, local repair, window
semantics, atomic multi-head admission and fail-closed unsupported inputs. Benchmarks distinguish raw samples, page capacity, reserved
GPU memory and latency; the dense Torch attention reference is not an
optimized FlashAttention throughput baseline.

Initial real Qwen3-8B qualification passes the strict FP32 comparison.
BF16 does **not** yet pass strict all-position token agreement, either
against HF eager or the separately declared dense FP32-accumulation oracle.
The original failures and their different rounding semantics must remain
visible; a successful cache-isolation test is not BF16 model qualification.
