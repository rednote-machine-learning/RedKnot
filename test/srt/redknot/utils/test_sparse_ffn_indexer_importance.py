"""Unit tests for RedKnot DeepSeek-V4 indexer-driven sparse-FFN selection.

These tests exercise the *real* selection methods from
``sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer`` by binding them to a
lightweight stub layer, so the production logic is verified directly without
loading the full model (CPU-only, no GPU / weights required).

What is covered:
  1. ``activation`` mode reproduces the legacy activation-L2-norm selection
     (backward compatibility).
  2. ``indexer`` mode reuses the native indexer ``c4_topk_lengths_raw`` signal:
     tokens whose indexer retrieved more context are kept; weakly-engaged
     tokens are dropped.
  3. ``blend`` mode reweights activation norm by the indexer mass.
  4. Robust fallback to ``activation`` when the indexer signal is unavailable
     (wrong compress_ratio, missing metadata, shape mismatch).
  5. ``recent_n`` always-keep and dense-prefix-layer behavior is preserved.

Run:
    python test/srt/redknot/test_sparse_ffn_indexer_importance.py
or with pytest:
    pytest -q test/srt/redknot/test_sparse_ffn_indexer_importance.py
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace

import torch

# Make the in-repo sglang importable when run directly.
_REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "python",
)
if _REPO_PYTHON not in sys.path:
    sys.path.insert(0, _REPO_PYTHON)


def _make_server_args(**overrides):
    args = SimpleNamespace(
        redknot_sparse_ffn_enable=True,
        redknot_sparse_ffn_dense_until=4,
        redknot_sparse_ffn_mass_thresh=0.6,
        redknot_sparse_ffn_deep_start=1000,  # keep mid-layer thresh
        redknot_sparse_ffn_mass_thresh_deep=0.1,
        redknot_sparse_ffn_recent_n=0,
        redknot_sparse_ffn_importance="activation",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class _StubLayer:
    """Minimal object exposing the attributes the real methods read.

    The real ``_redknot_indexer_token_importance`` /
    ``_select_redknot_sparse_ffn_tokens`` methods are attached at runtime (see
    ``setUpClass``) so the stub behaves like a real decoder layer for the
    selection logic, but without any of the heavy model construction.
    """

    def __init__(self, layer_id, compress_ratio, has_indexer, topk_lengths):
        self.layer_id = layer_id
        # compress_ratio / indexer live on the attention module (MQALayer),
        # mirrored here via a ``self_attn`` stub.
        self.self_attn = SimpleNamespace(
            compress_ratio=compress_ratio,
            indexer=object() if has_indexer else None,
        )
        self._topk_lengths = topk_lengths


class SparseFFNIndexerImportanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real methods and the module-level globals they use.
        from sglang.srt.models import deepseek_v4 as dv4

        cls.dv4 = dv4
        # Attach the *real* methods to the stub layer so the production code
        # path (including the internal self-call to the indexer-importance
        # helper) runs verbatim against our lightweight stub.
        _StubLayer._select_redknot_sparse_ffn_tokens = (
            dv4.DeepseekV4DecoderLayer._select_redknot_sparse_ffn_tokens
        )
        _StubLayer._redknot_indexer_token_importance = (
            dv4.DeepseekV4DecoderLayer._redknot_indexer_token_importance
        )

    def setUp(self):
        # Patch get_global_server_args + get_attn_backend on the module so the
        # bound methods see our stubs. Restored in tearDown.
        self._saved_gsa = self.dv4.get_global_server_args
        self._saved_gab = self.dv4.get_attn_backend
        self._server_args = _make_server_args()
        self._core_meta = SimpleNamespace(c4_topk_lengths_raw=None)
        self.dv4.get_global_server_args = lambda: self._server_args
        self.dv4.get_attn_backend = lambda: SimpleNamespace(
            forward_metadata=SimpleNamespace(core_attn_metadata=self._core_meta)
        )

    def tearDown(self):
        self.dv4.get_global_server_args = self._saved_gsa
        self.dv4.get_attn_backend = self._saved_gab

    def _run_select(self, layer, hidden_states):
        return layer._select_redknot_sparse_ffn_tokens(hidden_states)

    def _run_idx_imp(self, layer, num_tokens, device):
        return layer._redknot_indexer_token_importance(num_tokens, device)

    # ---- 1. activation mode == legacy behavior ----
    def test_activation_mode_matches_legacy(self):
        self._server_args.redknot_sparse_ffn_importance = "activation"
        self._server_args.redknot_sparse_ffn_mass_thresh = 0.5
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        # hidden states with clearly separated norms
        hs = torch.tensor(
            [[10.0, 0.0], [0.1, 0.0], [0.1, 0.0], [9.0, 0.0]], dtype=torch.float32
        )
        keep = self._run_select(layer, hs)
        self.assertIsNotNone(keep)
        # Largest-norm token (idx 0) must be kept; tiny tokens (1,2) dropped.
        self.assertTrue(bool(keep[0]))
        self.assertFalse(bool(keep[1]))
        self.assertFalse(bool(keep[2]))

    # ---- 2. indexer mode uses c4_topk_lengths_raw ----
    def test_indexer_mode_uses_indexer_signal(self):
        self._server_args.redknot_sparse_ffn_importance = "indexer"
        self._server_args.redknot_sparse_ffn_mass_thresh = 0.5
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        # Activation norm would pick token 3, but indexer says token 0 is the
        # globally-engaged one -> indexer must win.
        hs = torch.tensor(
            [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [99.0, 0.0]], dtype=torch.float32
        )
        self._core_meta.c4_topk_lengths_raw = torch.tensor(
            [500, 1, 1, 1], dtype=torch.int32
        )
        keep = self._run_select(layer, hs)
        self.assertIsNotNone(keep)
        # Token 0 dominates indexer mass -> kept; the high-activation token 3 is
        # NOT selected by indexer mass alone (mass_thresh 0.5, token0 ~ 0.99).
        self.assertTrue(bool(keep[0]))
        self.assertFalse(bool(keep[3]))

    # ---- 3. blend mode reweights activation by indexer mass ----
    def test_blend_mode_combines_signals(self):
        self._server_args.redknot_sparse_ffn_importance = "blend"
        self._server_args.redknot_sparse_ffn_mass_thresh = 0.5
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        hs = torch.tensor(
            [[8.0, 0.0], [8.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
        )
        # Indexer favors tokens 0 and 2; blend = act * idx_norm.
        self._core_meta.c4_topk_lengths_raw = torch.tensor(
            [400, 1, 400, 1], dtype=torch.int32
        )
        keep = self._run_select(layer, hs)
        self.assertIsNotNone(keep)
        # Token 0 (high act AND high idx) is the strongest -> kept.
        self.assertTrue(bool(keep[0]))

    # ---- 4a. fallback: indexer requested but wrong compress_ratio ----
    def test_fallback_when_not_c4_layer(self):
        self._server_args.redknot_sparse_ffn_importance = "indexer"
        layer = _StubLayer(10, compress_ratio=0, has_indexer=True, topk_lengths=None)
        self.assertIsNone(self._run_idx_imp(layer, 4, torch.device("cpu")))

    # ---- 4b. fallback: indexer length / token mismatch ----
    def test_fallback_on_shape_mismatch(self):
        self._server_args.redknot_sparse_ffn_importance = "indexer"
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        self._core_meta.c4_topk_lengths_raw = torch.tensor(
            [3, 3, 3],
            dtype=torch.int32,  # 3 != 4 tokens
        )
        imp = self._run_idx_imp(layer, 4, torch.device("cpu"))
        self.assertIsNone(imp)
        # Full select() then falls back to activation and still returns a mask.
        hs = torch.tensor(
            [[10.0, 0.0], [0.1, 0.0], [0.1, 0.0], [9.0, 0.0]], dtype=torch.float32
        )
        self._server_args.redknot_sparse_ffn_mass_thresh = 0.5
        keep = self._run_select(layer, hs)
        self.assertIsNotNone(keep)
        self.assertTrue(bool(keep[0]))

    # ---- 4c. fallback: missing metadata ----
    def test_fallback_when_metadata_missing(self):
        self._server_args.redknot_sparse_ffn_importance = "indexer"
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        self._core_meta.c4_topk_lengths_raw = None
        self.assertIsNone(self._run_idx_imp(layer, 4, torch.device("cpu")))

    # ---- 5a. dense prefix layers are never sparsified ----
    def test_dense_prefix_layer_returns_none(self):
        self._server_args.redknot_sparse_ffn_importance = "indexer"
        self._server_args.redknot_sparse_ffn_dense_until = 4
        layer = _StubLayer(2, compress_ratio=4, has_indexer=True, topk_lengths=None)
        hs = torch.randn(8, 4)
        self.assertIsNone(self._run_select(layer, hs))

    # ---- 5b. recent_n always kept ----
    def test_recent_n_always_kept(self):
        self._server_args.redknot_sparse_ffn_importance = "activation"
        self._server_args.redknot_sparse_ffn_mass_thresh = 0.01  # drop almost all
        self._server_args.redknot_sparse_ffn_recent_n = 2
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        hs = torch.tensor(
            [[5.0, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]], dtype=torch.float32
        )
        keep = self._run_select(layer, hs)
        self.assertIsNotNone(keep)
        # last two tokens forced kept regardless of importance
        self.assertTrue(bool(keep[-1]))
        self.assertTrue(bool(keep[-2]))

    # ---- 5c. disabled -> None ----
    def test_disabled_returns_none(self):
        self._server_args.redknot_sparse_ffn_enable = False
        layer = _StubLayer(10, compress_ratio=4, has_indexer=True, topk_lengths=None)
        self.assertIsNone(self._run_select(layer, torch.randn(8, 4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
