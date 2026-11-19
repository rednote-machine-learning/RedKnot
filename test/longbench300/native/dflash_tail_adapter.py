"""Unintegrated S2 prefill/DFlash tail adapter; no torch import or global patch.

This is a source-backed integration draft, NOT a GPU qualification. Installers
must still implement the integration contract below in an independent stage.
The current frozen S2 controller deliberately rejects DFlash and stays intact.

Integration contract
--------------------
* For an actual target layer, intercept precisely DecoderLayer._store_dflash_input
  in the S2 custom prefill path. Its argument is the attn_norm_forward result's
  residual.residual, BEFORE attention, often [1,n,hc_mult,hidden]. Call
  record_target with that tensor and the CPU-owned absolute query positions of
  that exact layer call. Never use final hidden, attention output, sampler140
  placeholders, or independently cached document hidden as a substitute.
* Use the unmodified native dflash_input_store as native_input_store. It performs
  the native FP32 mHC mean reduction followed by BF16 storage. No substitute
  reduction is implemented here. First3 uses the real original chunk positions;
  deferred deep groups use controller.ctx.positions_cpu, in the same row order.
* The installer must also handle the inactive ranks' target-layer calls: they
  still execute the existing EP schedule, but have no real tail to collect.
  This module owns only rank0's real-request feature/cache path. It is not an
  all-rank model wrapper and never skips collectives or placeholder sampling.
* After the FINAL deep flush, call finish with the actual final prefill metadata,
  native DFlash object and GPUCacheInterface. Earlier flushes may contribute
  final-window rows; they must NOT clear this collector. Native target features
  are reduced/copied while alive, so later mHC in-place mutation is harmless.
* Only speculative DECODE uses the normal original model, Worker, sampler and
  DFlash paths. Captures, ordinary Full, Prefix and all other modes must continue
  to call the original target store. This object does not hook those paths.

Why a tail can suffice: native DFlash only reads its SWA window and asserts that
it equals the main-model window. S1 retains document-tail128 plus all nondocument
rows, hence every final128 position is genuinely selected. We verify that set
in begin, keep the whole final window across chunk/group boundaries, and store
it at the native final position/state ring. This reasoning still needs GPU
feature/cache/rollback and same-policy output qualification, including partial
draft acceptance, final one-token chunks and request-slot reuse.

Pinned source anchors (relative to the sibling frozen s2-agent-v1 stage):
 NativeSnapshot/deepinfer/model_executor/models/decoder_layer.py:144-146
 NativeSnapshot/deepinfer/model_executor/layers/tilelang_ops/dflash.py:130-165
 NativeSnapshot/deepinfer/model_executor/models/mtp/dflash.py:198-228
 GatedEngram/kv_cache_engram/s1_plan.py:16-20
 GatedEngram/kv_cache_engram/s1_system_runtime.py:170-260
"""
from __future__ import annotations

from types import SimpleNamespace


class TailContractError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise TailContractError(message)


def ordered_positions(values, *, label):
    require(isinstance(values, (list, tuple)), label + " must be CPU list/tuple")
    values = tuple(values)
    require(all(type(v) is int and v >= 0 for v in values), label + " must contain nonnegative ints")
    require(all(a < b for a, b in zip(values, values[1:])), label + " must be strictly increasing")
    return values


class DFlashTailAdapter:
    """One real rank0 request, checkpoint-ordered target columns, native ops.

    torch_module and native_input_store are injected so CPU contract tests need
    neither torch nor GPU. Production must pass the pinned torch/native kernel.
    CPU tests do not certify kernel equivalence or actual distributed support.
    """

    def __init__(self, *, torch_module, native_input_store, target_layer_ids,
                 hidden_size, window_size=128):
        targets = tuple(target_layer_ids)
        require(targets and all(type(i) is int and 0 <= i < 40 for i in targets)
                and len(set(targets)) == len(targets), "unique checkpoint target layers required")
        require(type(hidden_size) is int and hidden_size > 0 and hidden_size % 4 == 0,
                "native feature stride must be divisible by four")
        require(window_size == 128, "draft supports only the unchanged S1 window128 policy")
        require(callable(native_input_store), "actual native feature reduction is required")
        self.t, self.native_input_store = torch_module, native_input_store
        self.targets, self.hidden_size, self.window_size = targets, hidden_size, window_size
        self.state, self.buffer = "IDLE", None
        self.coverage, self.events = {}, []

    def begin(self, *, request_id, phase, input_tokens, selected_positions_cpu):
        require(self.state in ("IDLE", "COMMITTED"), "previous tail request not committed/aborted")
        require(isinstance(request_id, str) and request_id and isinstance(phase, str) and phase,
                "request and phase identities required")
        require(type(input_tokens) is int and input_tokens > 0, "positive input length required")
        selected = ordered_positions(selected_positions_cpu, label="selected positions")
        require(selected and selected[-1] < input_tokens, "selected position outside input")
        start = max(0, input_tokens-self.window_size)
        expected = tuple(range(start, input_tokens))
        require(set(expected).issubset(selected), "final DFlash window contains uncomputed S1 rows")
        self.request_id, self.phase, self.input_tokens = request_id, phase, input_tokens
        self.start, self.positions, self.selected = start, expected, selected
        self.buffer, self.coverage, self.events = None, {i: set() for i in self.targets}, []
        self.state = "COLLECTING"

    def record_target(self, *, request_id, phase, layer_id, attention_input,
                      positions_cpu, source="native_decoder_attention_input"):
        """Consume the real native _store_dflash_input argument while alive."""
        require(self.state == "COLLECTING", "tail is not collecting")
        try:
            require((request_id, phase) == (self.request_id, self.phase), "foreign target request/phase")
            require(layer_id in self.targets, "layer is not a checkpoint DFlash target")
            require(source == "native_decoder_attention_input", "wrong feature source")
            positions = ordered_positions(positions_cpu, label="actual layer positions")
            require(not positions or positions[-1] < self.input_tokens, "layer position outside request")
            shape = tuple(attention_input.shape)
            require(len(shape) in (2, 3, 4) and shape[-1] == self.hidden_size,
                    "native attention feature shape differs")
            if len(shape) == 4:
                require(shape[0] == 1, "S1 mHC batch dimension must be one")
                rows, hc_mult = shape[1], shape[2]
            elif len(shape) == 3:
                rows, hc_mult = shape[0], shape[1]
            else:
                rows, hc_mult = shape[0], 1
            require(rows == len(positions) and hc_mult > 0, "feature/absolute-position rows differ")
            require(attention_input.dtype == self.t.bfloat16 and attention_input.is_contiguous(),
                    "native attention input must be contiguous BF16")
            selected = [(i, p) for i, p in enumerate(positions) if p >= self.start]
            if not selected:
                return 0
            indices, absolute = zip(*selected)
            require(tuple(indices) == tuple(range(indices[0], indices[-1]+1))
                    and tuple(absolute) == tuple(range(absolute[0], absolute[-1]+1)),
                    "a final-window layer slice is not contiguous")
            require(not (set(absolute) & self.coverage[layer_id]), "duplicate target/window write")
            if self.buffer is None:
                self.buffer = attention_input.new_empty((len(self.positions), self.hidden_size*len(self.targets)))
            require(self.buffer.device == attention_input.device and self.buffer.dtype == attention_input.dtype,
                    "target feature device/dtype changed")
            # Preserve the actual native reduction, including mHC mean/cast.
            feature = attention_input.view(rows, hc_mult, self.hidden_size).narrow(0, indices[0], len(indices))
            target_column = self.targets.index(layer_id)*self.hidden_size
            output = self.buffer.narrow(0, absolute[0]-self.start, len(absolute)).narrow(1, target_column, self.hidden_size)
            self.native_input_store(feature, output)
            self.coverage[layer_id].update(absolute)
            self.events.append(dict(layer=layer_id, start=absolute[0], end=absolute[-1]+1,
                                    rows=len(absolute), native_hc_mult=hc_mult))
            return len(absolute)
        except BaseException:
            self.state = "FAILED"
            raise

    def finish(self, *, request_id, phase, final_metadata, native_dflash, kv_caches):
        """Populate native DFlash main SWA once, before first actual decode.

        Metadata is a minimal read-only input to native update_main_cache, which
        consumes positions, mirror_cp_reverse_index and slot_mapping_store_swa.
        Ring addressing derives from the FINAL REAL prefill slot, never a CPU
        guessed state ID. Device assertions do not read tensor values to host.
        """
        require(self.state == "COLLECTING", "tail cannot commit in this state")
        try:
            require((request_id, phase) == (self.request_id, self.phase), "foreign final request/phase")
            expected = set(self.positions)
            require(self.buffer is not None and all(v == expected for v in self.coverage.values()),
                    "incomplete target-layer tail features")
            require(tuple(native_dflash.target_layer_ids) == self.targets
                    and native_dflash.window_size == self.window_size,
                    "actual DFlash target order/window differs")
            require(callable(native_dflash.update_main_cache) and len(native_dflash.layers) > 0,
                    "native DFlash main-cache interface missing")
            m = final_metadata
            marker = getattr(m, "canonical_request_binding", None)
            require(isinstance(marker, dict) and marker.get("request_id") == request_id
                    and marker.get("phase") == phase and marker.get("end") == self.input_tokens,
                    "metadata is not the bound final prefill")
            require(m.is_prompt is True and m.num_reqs == 1 and m.mirror_cp_reverse_index is None,
                    "only actual CP1 single-request prefill metadata is supported")
            require(m.positions.ndim == 1 and m.positions.shape[0] > 0
                    and m.slot_mapping_store_swa.shape == m.positions.shape
                    and tuple(m.state_block_idx.shape) == (1,), "final native slot geometry differs")
            require(m.positions.device == self.buffer.device
                    and m.slot_mapping_store_swa.device == self.buffer.device
                    and m.state_block_idx.device == self.buffer.device, "metadata device differs")
            caches = [kv_caches[layer.layer_idx].swa_cache for layer in native_dflash.layers]
            require(all(c is not None and c.ndim >= 2 and c.device == self.buffer.device for c in caches),
                    "DFlash SWA cache missing or on another device")
            ring = caches[0].shape[1]
            require(ring >= self.window_size and all(c.shape[1] == ring for c in caches),
                    "DFlash SWA ring layout differs across layers")
            # Retain asynchronous device-side checks used by the native S1 path.
            state, last_slot = m.state_block_idx[0], m.slot_mapping_store_swa[-1]
            self.t._assert_async(m.positions[-1] == self.input_tokens-1, "not the final prompt position")
            self.t._assert_async((last_slot >= 0) & (last_slot // ring == state), "final slot/state ownership differs")
            for c in caches:
                self.t._assert_async((state > 0) & (state < c.shape[0]), "DFlash state is placeholder or out of bounds")
            positions = self.t.arange(self.start, self.input_tokens, dtype=m.positions.dtype, device=m.positions.device)
            base = last_slot - ((self.input_tokens-1) % ring)
            slots = (base + positions % ring).to(dtype=m.slot_mapping_store_swa.dtype)
            tail_meta = SimpleNamespace(positions=positions, slot_mapping_store_swa=slots,
                                       mirror_cp_reverse_index=None)
            native_dflash.update_main_cache(self.buffer, tail_meta, kv_caches)
            self.state = "COMMITTED"
            return dict(schema="s2-dflash-tail-contract-draft-v1", request_id=request_id, phase=phase,
                        input_tokens=self.input_tokens, window_size=self.window_size,
                        target_layer_ids=list(self.targets), retained_positions=list(self.positions),
                        feature_shape=list(self.buffer.shape), coverage_by_target={str(k):len(v) for k,v in self.coverage.items()},
                        collection_events=list(self.events), native_main_cache_update_calls=1,
                        selected_positions_unchanged=True, sampler_placeholder_features_used=False,
                        kernel_implementation="injected_native_dflash_input_store_and_update_main_cache",
                        integrated_all_rank_runtime=False, gpu_qualified=False)
        except BaseException:
            self.state = "FAILED"
            raise

    def abort(self):
        """Drop only owned temporary feature storage; never touches native cache."""
        self.buffer, self.coverage, self.events, self.state = None, {}, [], "IDLE"
