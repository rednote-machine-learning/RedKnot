"""Owned S1 prefill integration of native DFlash target features.

Only policy prefill intercepts a None output buffer. Every native mode and
native decode, including graph capture, retain the original store operation.
Qualification hashes the real native-reduced features and valid main-cache
rows before any speculative decode can overwrite them. CPU tests certify the
routing/coverage contract only; actual CUDA parity remains a separate gate.
"""
from __future__ import annotations

from contextlib import contextmanager

from .core import require
from .index_hit20 import fingerprint
from dflash_tail_adapter import DFlashTailAdapter


class S1DFlashBridge:
    def __init__(self, owner, *, native_input_store=None):
        self.owner, self.frame = owner, None
        indexed = [(layer.dflash_input_index, layer) for layer in owner.layers
                   if getattr(layer, "dflash_input_index", None) is not None]
        indexed.sort(key=lambda item: item[0])
        require(indexed and [i for i, _ in indexed] == list(range(len(indexed))),
                "native DFlash target indices are missing or duplicated")
        self.target_layers = tuple(layer for _, layer in indexed)
        targets = tuple(layer.layer_idx for layer in self.target_layers)
        model = owner.native_model
        require(model.dflash_input_size == model.hidden_size * len(targets),
                "native DFlash target feature width differs")
        if native_input_store is None:
            from deepinfer.model_executor.layers.tilelang_ops.dflash import dflash_input_store
            native_input_store = dflash_input_store
        self.collector = DFlashTailAdapter(torch_module=owner.t,
            native_input_store=native_input_store, target_layer_ids=targets,
            hidden_size=model.hidden_size, window_size=128)
        self.receipt = None
        self.reference = None

    def install(self):
        for layer in self.target_layers:
            original = layer._store_dflash_input
            bridge = self
            def store(module, hidden_states, dflash_input, _original=original):
                if dflash_input is not None:
                    return _original(hidden_states, dflash_input)
                require(bridge.owner.mode in ("reference_sparse", "sensitive")
                        and bridge.frame is not None and bridge.frame[0] == module.layer_idx,
                        "DFlash feature store has no owned policy layer frame")
                bridge.collector.record_target(request_id=bridge.owner.rid,
                    phase=bridge.owner.phase, layer_id=module.layer_idx,
                    attention_input=hidden_states, positions_cpu=bridge.frame[1])
                return None
            self.owner._patch(layer, "_store_dflash_input", store)
        return self

    def check_native(self, dflash):
        require(dflash is not None
                and tuple(dflash.target_layer_ids) == self.collector.targets
                and dflash.window_size == self.collector.window_size,
                "actual DFlash object differs from installed target/window contract")

    @contextmanager
    def layer_call(self, layer_id, positions):
        if layer_id not in self.collector.targets:
            yield
            return
        require(self.frame is None, "overlapping DFlash target layer frames")
        self.frame = (layer_id, tuple(positions))
        try:
            yield
        finally:
            self.frame = None

    def reset(self):
        require(self.frame is None, "DFlash reset during target execution")
        self.collector.abort()
        self.receipt = None
        o = self.owner
        if o.mode == "reference_sparse":
            # Base reset has already required an empty prompt-specific sparse
            # reference. S2 begin_request preserves banks but starts a fresh one.
            self.clear_reference()
        if o.mode in ("reference_sparse", "sensitive"):
            selected = [p for _, prepared in sorted(o.prepared.items())
                        for p in prepared["plan"]["selected_positions"]]
            self.collector.begin(request_id=o.rid, phase=o.phase,
                                 input_tokens=len(o.ids), selected_positions_cpu=selected)

    def finish(self, meta, dflash, caches):
        o = self.owner
        self.check_native(dflash)
        require(self.receipt is None, "DFlash tail was already committed")
        receipt = self.collector.finish(request_id=o.rid, phase=o.phase,
            final_metadata=meta, native_dflash=dflash, kv_caches=caches)
        receipt.update(schema="s2-dflash-tail-runtime-v1", integrated_all_rank_runtime=True,
            input_ids_sha256=o.input_sha, native_dflash_layer_ids=[layer.layer_idx for layer in dflash.layers],
            state_id=None, state_id_verified_on_device=True, qualification_fingerprints=None,
            same_policy_reference_exact=False, fingerprint_before_decode=True)
        if o._qualifying_sparse():
            with o.allow_sync():
                state = int(meta.state_block_idx[0].item())
                positions = o.t.arange(self.collector.start, self.collector.input_tokens,
                                       dtype=o.t.long, device=meta.positions.device)
                cache_fingerprints = []
                for layer in dflash.layers:
                    cache = caches[layer.layer_idx].swa_cache
                    valid = cache[state].index_select(0, positions % cache.shape[1])
                    cache_fingerprints.append(dict(layer=layer.layer_idx, value=fingerprint(valid, o.t)))
                actual = dict(target_features=fingerprint(self.collector.buffer, o.t),
                              valid_main_swa=cache_fingerprints)
            receipt.update(state_id=state, qualification_fingerprints=actual)
            if o.mode == "reference_sparse":
                require(self.reference is None, "DFlash same-policy reference was not cleared")
                self.reference = (o.input_sha, self.collector.targets, tuple(self.collector.positions), actual)
            else:
                require(self.reference == (o.input_sha, self.collector.targets,
                                          tuple(self.collector.positions), actual),
                        "S1 DFlash real tail features/main-cache byte parity failed")
                receipt["same_policy_reference_exact"] = True
        self.receipt = receipt
        # No native cache allocation is replaced. Only our temporary feature
        # matrix is released, after native main-cache initialization completes.
        self.collector.buffer = None

    def report(self):
        if self.owner.mode not in ("reference_sparse", "sensitive"):
            return dict(schema="s2-dflash-tail-runtime-v1", enabled=False,
                        original_native_dflash_path=True,
                        target_layer_ids=list(self.collector.targets))
        require(self.receipt is not None and self.collector.state == "COMMITTED",
                "S1 policy did not initialize the final DFlash main cache")
        return dict(self.receipt, enabled=True)

    def clear_reference(self):
        self.reference = None

    def close(self):
        require(self.frame is None, "DFlash close during target execution")
        self.collector.abort()
