# Copyright 2024-2026 SGLang RedKnot Integration.
"""SegPagedAttention backend for RedKnot.

This backend exposes SegPagedAttention as a first-class SGLang attention
backend via ``--attention-backend segpaged``.

Design, following the RedKnot paper
----------------------------------
SegPagedAttention is the physical/runtime substrate for DuoAttention-style
head heterogeneity:

- Global KV heads keep full-context pages.
- Local KV heads keep only ``sink + recent window`` pages.
- Attention executes over each head's real sequence length via varlen FA
  (or an exact PyTorch fallback when FA-3 is unavailable).

The implementation intentionally reuses :class:`RedKnotAttnBackend` for the
prefill/extend control flow and forces the decode path to use the per-head
SegPaged view. This keeps the API small while exposing a distinct backend
name for experiments and tests.

CLI
---

.. code-block:: bash

   python -m sglang.launch_server \
     --attention-backend segpaged \
     --segpaged-head-config-path /path/to/head_config.json \
     --segpaged-page-size 64

If ``--segpaged-head-config-path`` is omitted, the backend falls back to
``--redknot-head-config-path`` for convenience.
"""

from __future__ import annotations

import logging
from typing import Optional

from sglang.srt.layers.attention.redknot.head_config import HeadClassConfig
from sglang.srt.layers.attention.redknot.ops_flash import DEFAULT_Q_CHUNK
from sglang.srt.layers.attention.redknot_backend import RedKnotAttnBackend

logger = logging.getLogger(__name__)


class SegPagedAttnBackend(RedKnotAttnBackend):
    """RedKnot backend variant with SegPagedAttention decode enabled."""

    def __init__(
        self,
        model_runner,
        head_config: Optional[HeadClassConfig] = None,
        q_chunk_size: int = DEFAULT_Q_CHUNK,
        kernel: str = "fa2",
        page_size: int = 64,
    ):
        if head_config is None:
            head_config = _load_segpaged_head_config(model_runner)
        super().__init__(
            model_runner,
            head_config=head_config,
            q_chunk_size=q_chunk_size,
            kernel=kernel,
            use_segpaged_decode=True,
            segpaged_page_size=page_size,
        )
        logger.info(
            "SegPagedAttnBackend initialized (kernel=%s, page_size=%d, head_config=%s)",
            kernel,
            page_size,
            "yes" if self.head_config is not None else "no",
        )


def _load_segpaged_head_config(model_runner) -> Optional[HeadClassConfig]:
    server_args = getattr(model_runner, "server_args", None)
    if server_args is None:
        return None
    path = getattr(server_args, "segpaged_head_config_path", None)
    if not path:
        path = getattr(server_args, "redknot_head_config_path", None)
    if not path:
        return None
    try:
        cfg = HeadClassConfig.from_json(path)
        logger.info(
            "SegPagedAttention: loaded head config %s (L=%d, KVH=%d, summary=%s)",
            path,
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.summary(),
        )
        return cfg
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "SegPagedAttention: failed to load head config %s: %s", path, exc
        )
        return None


__all__ = ["SegPagedAttnBackend"]
