# Head-paged KV ownership and sharing

This package implements an experimental bounded MHA/GQA KV manager and a
direct paged attention backend. Pages belong to physical KV heads/groups,
not to query heads. The existing RedKnot `segpaged_attention` entry point
dispatches to this backend when given `ManagedSegPagedKVCache`.

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

This change provides a callable managed SegPaged path, not automatic
replacement of the SGLang scheduler's existing request/KV pools. It does
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
  test/srt/redknot/test_head_kv_transfer.py
python test/srt/redknot/benchmark_head_kv_manager.py \
  --device cuda:0 --dtype bfloat16 --output benchmark.json
python test/srt/redknot/validate_head_kv_qwen3.py \
  --model-path /path/to/Qwen3-8B --device cuda --dtype bfloat16 \
  --output qwen3.json
```

Tests cover local ownership/COW/events, direct attention, real HTTP
transfers and faults, restart decisions, migration fencing and snapshot
validation. Benchmarks distinguish raw samples, page capacity, reserved
GPU memory and latency; the dense Torch attention reference is not an
optimized FlashAttention throughput baseline.

Initial real Qwen3-8B qualification passes the strict FP32 comparison.
BF16 does **not** yet pass strict all-position token agreement, either
against HF eager or the separately declared dense FP32-accumulation oracle.
The original failures and their different rounding semantics must remain
visible; a successful cache-isolation test is not BF16 model qualification.
