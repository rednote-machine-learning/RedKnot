"""Contract: configuration -> validated policy; catch silent approximation/drift.

The cheapest boundary is a stdlib unit test: no model, torch, server or GPU.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_redknot.config import (
    import_sglang_head_policy,
    load_config,
    policy_fingerprint,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def base(self):
        return {"model_revision": "model-revision-123", "local_heads": {"3": [1, 0]}}

    def test_unset_is_inert(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_config())

    def test_explicit_policy_is_normalized_and_approximation_is_off(self):
        config = validate_config(self.base())
        self.assertEqual(config["engine_family"], "mha")
        self.assertEqual(config["local_heads"], {"3": [0, 1]})
        self.assertFalse(config["allow_approximate"])
        self.assertEqual(policy_fingerprint(config), policy_fingerprint(self.base()))

    def test_rejects_silent_invalid_settings(self):
        for change in (
            {"schema_version": True},
            {"boundary_tokens": True},
            {"max_cache_bytes": -1},
            {"allow_approximate": "false"},
            {"rotary_dim": 3},
            {"rope_theta": float("nan")},
            {"rope_theta": False},
            {"unexpected": 7},
            {"local_heads": {"2": [1, 1]}},
            {"local_heads": {"2": [True]}},
            {"model_revision": ""},
            {"supported_vllm_commit": "main"},
            {"engine_family": "deepseek_v4"},
            {"engine_family": None},
            {"engine_family": []},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**self.base(), **change})

    def test_dsv4_family_uses_logical_query_heads_and_native_rope(self):
        config = validate_config(
            {
                **self.base(),
                "engine_family": "deepseek_v4_flash",
                "local_heads": {"3": [63, 0, 17]},
            }
        )
        self.assertEqual(config["local_heads"], {"3": [0, 17, 63]})
        self.assertEqual(config["rope_theta"], 10000.0)  # Unused compatibility value.
        self.assertIsNone(config["rotary_dim"])
        self.assertEqual(validate_config(config), config)

    def test_dsv4_refuses_overrides_that_would_not_control_native_rope(self):
        for change in ({"rope_theta": 50000}, {"rotary_dim": 64}):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "native per-layer RoPE"),
            ):
                validate_config(
                    {
                        **self.base(),
                        "engine_family": "deepseek_v4_flash",
                        **change,
                    }
                )

    def test_engine_families_have_distinct_fingerprints(self):
        self.assertNotEqual(
            policy_fingerprint(self.base()),
            policy_fingerprint({**self.base(), "engine_family": "deepseek_v4_flash"}),
        )

    def test_json_duplicate_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for text in ('{"enabled":false,"enabled":true}', '{"rope_theta":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_source_windows_and_retrieval_are_not_implicitly_ported(self):
        config = import_sglang_head_policy(
            {
                "num_layers": 3,
                "num_kv_heads": 2,
                "dense_prefix_layers": 1,
                "kv_head_classification": [
                    ["local", "global"],
                    ["local", "retrieval"],
                    ["dense", "global"],
                ],
            },
            model_revision="pinned",
            rope_theta=1000000,
        )
        self.assertEqual(config["local_heads"], {"1": [0]})
        self.assertFalse(config["allow_approximate"])

    def test_identity_changes_with_model_and_policy(self):
        first = policy_fingerprint(self.base())
        self.assertNotEqual(
            first,
            policy_fingerprint(
                {
                    **self.base(),
                    "model_revision": "other",
                }
            ),
        )
        self.assertNotEqual(
            first,
            policy_fingerprint(
                {
                    **self.base(),
                    "boundary_tokens": 64,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
