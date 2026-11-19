"""Optional reuse of immutable S1 compressed pre-norm global bases.

Importing this module does not import torch, DeepInfer or the frozen package.
The caller supplies its existing cache class and an actual bank generation.
Every miss uses that class's original begin_phase implementation.  Hits skip
only repeated offline materialization/compression; online overlays, native
stores and compressor carry remain owned by the original implementation.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext

GLOBAL_SOURCES = (8, 14, 20)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _function_identity(value):
    return id(getattr(value, '__func__', value))


def _tensor_signature(value):
    """Host metadata only: no tensor values, transfers or device synchronizes."""
    return (id(value), int(value._version), tuple(value.shape),
            str(value.dtype), str(value.device))


def _module_signature(module):
    # Native Compressor.forward_pre_norm consumes the preprojected tensor and
    # compression ratio. Parameters/buffers are included conservatively so
    # future native implementations cannot reuse a base after weight changes.
    scalars = tuple(sorted((name, value) for name, value in vars(module).items()
        if not name.startswith('_') and type(value) in (str, int, bool, float, type(None))))
    tensors = []
    for method in ('named_parameters', 'named_buffers'):
        iterate = getattr(module, method, None)
        if iterate is not None:
            tensors.extend((method, name, _tensor_signature(value)) for name, value in iterate())
    forward = getattr(module, 'forward_pre_norm', None)
    return (id(module), type(module).__module__, type(module).__qualname__,
            _function_identity(forward), scalars, tuple(tensors))


def cached_global_base_class(base_class):
    """Return a private subclass; never mutate an imported class/module.

    Instantiate as usual, then configure once, for example::

        cache.configure_global_base_reuse(
            enabled=True, bank_generation=lambda: controller.bank_generation)

    The callback is read on each phase. Only one exact layout is retained, so
    changing layouts releases the previous cached base rather than growing an
    unbounded content cache. An integer may be supplied when the caller updates
    it explicitly on every bank generation change.
    """
    class GlobalBaseReuseCache(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._global_base_reuse_enabled = False
            self._global_base_generation = None
            self._retained_global_base = None
            self._global_base_counts = dict(hits=0, misses=0, invalidations=0,
                disabled_phases=0, nonbuilding_phases=0, versioned_base_clones=0)
            self._global_base_last = None

        def configure_global_base_reuse(self, *, enabled, bank_generation=None,
                                        retain_cached_base=False):
            if type(enabled) is not bool:
                raise ValueError('global-base reuse enabled must be bool')
            if type(retain_cached_base) is not bool:
                raise ValueError('retain_cached_base must be bool')
            if enabled and not (callable(bank_generation) or type(bank_generation) is int):
                raise ValueError('enabled global-base reuse requires the actual bank generation')
            self._global_base_reuse_enabled = enabled
            self._global_base_generation = bank_generation
            # Optional paired timing: a control request performs the original
            # build while retaining a separate read-only candidate for the
            # next optimized request. It is never read by the disabled path.
            if not enabled and not retain_cached_base:
                self._retained_global_base = None

        def invalidate_global_base_reuse(self):
            """Optional explicit lifetime hook, with no bank or native mutation."""
            if self._retained_global_base is not None:
                self._global_base_counts['invalidations'] += 1
            self._retained_global_base = None

        def _global_base_key(self):
            generation = self._global_base_generation
            generation = generation() if callable(generation) else generation
            if type(generation) is not int or generation < 0:
                raise ValueError('actual bank generation must be a nonnegative integer')
            documents = tuple((d['key'], d['start'], d['end']) for d in self.docs)
            entries = []
            for d in self.docs:
                for layer in GLOBAL_SOURCES:
                    # Preserve original complete/scope/version bank validation.
                    tensor = self.entry(d['key'], 'global_raw', layer)
                    entry = self.bank[d['key'], 'global_raw', layer]
                    entries.append((d['key'], layer, entry['length'], entry['local_start'],
                                    entry['scope'], _tensor_signature(tensor)))
            compressors = tuple((layer, int(self.layers[layer].self_attn.kv_compress_ratio),
                _module_signature(self.layers[layer].self_attn.PackedWkv)) for layer in GLOBAL_SOURCES)
            return (self.total, documents, _digest(self.scope), generation,
                    str(self.device), tuple(entries), compressors, id(self.ops),
                    _function_identity(self.ops.compress_canonical))

        def _versioned_bases(self):
            bases, copied = dict(self.global_bases), False
            for layer, tensor in list(bases.items()):
                try:
                    tensor._version
                except RuntimeError:
                    # Native phase execution can run under inference_mode.
                    # Its transient outputs then have no version counter. Own
                    # an equal normal tensor once, so accidental in-place
                    # writes remain detectable without hashing GPU contents.
                    value = self._allocate(tensor.shape[0], tensor.shape[1],
                                           tensor.dtype, tensor.device)
                    with self.t.inference_mode(False), self.t.no_grad():
                        value.copy_(tensor)
                    bases[layer], copied = value, True
                    self._global_base_counts['versioned_base_clones'] += 1
            if copied and self.device.type == 'cuda':
                # The frozen cold path synchronizes before returning. Keep
                # that readiness promise for the one-time owned copies too.
                with self.allow_sync() if self.allow_sync is not None else nullcontext():
                    self.t.cuda.synchronize(self.device)
            self.global_bases = dict(bases)
            return bases

        def begin_phase(self, phase, *, build_global_base=True):
            if not self._global_base_reuse_enabled:
                self._global_base_counts['disabled_phases'] += 1
                self._global_base_last = dict(phase=phase, route='disabled')
                return super().begin_phase(phase, build_global_base=build_global_base)
            if not build_global_base:
                self._global_base_counts['nonbuilding_phases'] += 1
                self._global_base_last = dict(phase=phase, route='nonbuilding')
                return super().begin_phase(phase, build_global_base=False)

            started = time.perf_counter()
            key = self._global_base_key()
            retained = self._retained_global_base
            valid = retained is not None and retained['key'] == key
            if valid:
                valid = all(_tensor_signature(value) == retained['signatures'][layer]
                            for layer, value in retained['bases'].items())
            if valid:
                # This keeps all original phase resets, including a subclass's
                # phase enable flag, carry/events, next_chunk and phase counter.
                super().begin_phase(phase, build_global_base=False)
                self.global_bases = dict(retained['bases'])
                self._global_base_counts['hits'] += 1
                route = 'hit'
            else:
                if retained is not None:
                    self._global_base_counts['invalidations'] += 1
                self._retained_global_base = None
                super().begin_phase(phase, build_global_base=True)
                bases = self._versioned_bases()
                if tuple(sorted(bases)) != GLOBAL_SOURCES:
                    raise ValueError('native global base sources changed')
                self._retained_global_base = dict(key=key, bases=bases,
                    signatures={layer: _tensor_signature(value) for layer, value in bases.items()})
                self._global_base_counts['misses'] += 1
                route = 'miss'
            # Include key validation and reset work, never hide it from TTFT.
            self.preparation_s = time.perf_counter() - started
            self._global_base_last = dict(phase=phase, route=route, key_sha256=_digest(key),
                preparation_s=self.preparation_s,
                retained_payload_bytes=sum(v.numel()*v.element_size() for v in self.global_bases.values()),
                original_online_overlay_and_native_store_unchanged=True,
                cached_representation='compressed_prenorm_bf16',
                physical_speedup_measured=False)
            return self.preparation_s

        def report(self):
            result = super().report()
            result['global_base_reuse'] = dict(schema='rag-global-base-reuse-v1',
                enabled=self._global_base_reuse_enabled, counts=dict(self._global_base_counts),
                maximum_retained_layouts=1,
                retained_layouts=int(self._retained_global_base is not None),
                last_phase=None if self._global_base_last is None else dict(self._global_base_last))
            return result

    GlobalBaseReuseCache.__name__ = 'GlobalBaseReuse' + base_class.__name__
    GlobalBaseReuseCache.__qualname__ = GlobalBaseReuseCache.__name__
    return GlobalBaseReuseCache
