"""Attention computation workflow in RedKnot for DeepSeek-V4.1-Flash.

Projection, attention, cache storage and decode remain inherited. The current
adapter restores canonical SWA rows after native preparation, so it still
executes the native KV projection; this is not a projection-skip optimization.
"""
from deepinfer.model_executor.models.v4_modules.attn import V4Attention


class RedKnotV4Attention(V4Attention):
    """Restore request-bound SWA rows without copying native ``forward``."""

    _redknot_swa_adapter_version = 1

    def _prepare_swa_kv(self, kv_swa, swa_cache, input_metadata, use_cp):
        marker = getattr(input_metadata, "transparent_swa_binding", None)
        if marker is None:
            return super()._prepare_swa_kv(kv_swa, swa_cache, input_metadata, use_cp)
        if not isinstance(marker, dict) or marker.get("session") is None:
            raise ValueError("invalid RedKnot SWA request binding")
        session = marker["session"]
        layer = self.layer_idx
        if layer not in session.supported_layers:
            return super()._prepare_swa_kv(kv_swa, swa_cache, input_metadata, use_cp)

        state = session.state
        if (session.active is not True or marker is not state["marker"]
                or marker.get("request_id") != state["request_id"]
                or input_metadata.is_prompt is not True or use_cp):
            raise ValueError("RedKnot SWA request ownership mismatch")
        start, end = marker["start"], marker["end"]
        if state["covered"][layer] != start:
            raise ValueError("SWA layer coverage is missing, repeated or out of order")
        session.ops.validate_raw(kv_swa, end-start)
        session.ops.validate_frame(input_metadata, start, end)
        canonical = session.ops.canonical(self, kv_swa) if session.capture else None

        result = super()._prepare_swa_kv(kv_swa, swa_cache, input_metadata, use_cp)
        if (not isinstance(result, tuple) or len(result) != 2
                or result[0] is None or result[1] is not None):
            raise ValueError("native CP1 prefill SWA contract changed")
        prepared, wait = result
        if session.capture:
            state["captured"][layer].append(canonical)
        else:
            for left, right, _chunk_id, cache in session.matches:
                lo, hi = max(start, left+session.boundary_tokens), min(end, right)
                if lo >= hi:
                    continue
                session.ops.replace(
                    self, prepared, cache.layers[layer][lo-left:hi-left],
                    lo-start, hi-start, input_metadata.positions[lo-start:hi-start])
                state["reused"][layer] += hi-lo
        state["covered"][layer] = end
        # Inherited forward consumes these rows for attention and persists them
        # through store_kv_cache, preserving the native decode cache lifecycle.
        return prepared, wait
