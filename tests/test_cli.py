"""Launch I/O contracts without importing vLLM or touching GPUs/network.

Guard against implicit downloads, unsupported architectures and unsafe launch
defaults using temporary local-model fixtures; command construction is enough.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_redknot.cli import doctor, installed_plugin, serve_command
from vllm_redknot.dsv4_runner import FLASH_REVISION


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "local model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"architectures": ["Qwen3ForCausalLM"]})
        )
        self.config = self.root / "policy.json"
        self.settings = {
            "model_revision": "sha256:verified-fixture",
            "local_heads": {"1": [0]},
        }
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.settings))

    def test_launch_uses_existing_path_offline_and_isolated_defaults(self):
        with patch.dict(os.environ, {"VLLM_PLUGINS": "other"}, clear=True):
            command, environment = serve_command(str(self.model), str(self.config))
        self.assertIn(str(self.model.resolve()), command)
        self.assertIn("--enforce-eager", command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertIn("--no-enable-chunked-prefill", command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["VLLM_PLUGINS"], "redknot")
        self.assertEqual(environment["VLLM_USE_V2_MODEL_RUNNER"], "0")
        self.assertEqual(
            json.loads(command[command.index("--attention-config") + 1]),
            {"backend": "CUSTOM"},
        )

    def test_model_identifier_does_not_trigger_download(self):
        with self.assertRaisesRegex(ValueError, "existing local checkpoint"):
            serve_command("not-a-local-model/remote-id", str(self.config))

    def test_deepseek_is_rejected_instead_of_silently_using_mha(self):
        (self.model / "config.json").write_text(
            json.dumps({"architectures": ["DeepseekV4ForCausalLM"]})
        )
        with self.assertRaisesRegex(ValueError, "explicit engine_family"):
            serve_command(str(self.model), str(self.config))

    def flash_config(self):
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DeepseekV4ForCausalLM"],
                    "num_hidden_layers": 43,
                    "num_attention_heads": 64,
                    "head_dim": 512,
                    "qk_rope_head_dim": 64,
                    "o_groups": 8,
                }
            )
        )
        self.settings.update(
            engine_family="deepseek_v4_flash",
            model_revision=FLASH_REVISION,
            local_heads={"3": [1, 7, 63]},
        )
        self.write_config()

    def test_explicit_flash_uses_native_mla_backend_and_logical_query_heads(self):
        self.flash_config()
        command, environment = serve_command(str(self.model), str(self.config))
        self.assertEqual(
            json.loads(command[command.index("--attention-config") + 1]),
            {"backend": "FLASHMLA_SPARSE_DSV4"},
        )
        self.assertNotIn("CUSTOM", " ".join(command))
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["VLLM_USE_V2_MODEL_RUNNER"], "0")

    def test_flash_rejects_wrong_precision_revision_and_head_geometry(self):
        self.flash_config()
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            serve_command(str(self.model), str(self.config), dtype="float16")
        self.settings["model_revision"] = "unverified-checkpoint"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "pinned Flash-0731"):
            serve_command(str(self.model), str(self.config))
        self.flash_config()
        self.settings["local_heads"] = {"3": [64]}
        self.write_config()
        with self.assertRaises(ValueError):
            serve_command(str(self.model), str(self.config))
        self.flash_config()
        checkpoint = json.loads((self.model / "config.json").read_text())
        checkpoint["num_attention_heads"] = 128
        (self.model / "config.json").write_text(json.dumps(checkpoint))
        with self.assertRaisesRegex(ValueError, "64-query-head"):
            serve_command(str(self.model), str(self.config))

    def test_flash_family_cannot_silently_route_an_mha_checkpoint(self):
        self.settings["engine_family"] = "deepseek_v4_flash"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "DeepseekV4ForCausalLM"):
            serve_command(str(self.model), str(self.config))

    def test_doctor_checks_selected_family_without_claiming_gpu_validation(self):
        self.flash_config()
        with (
            patch("vllm_redknot.cli.installed_plugin", return_value=True),
            patch(
                "vllm_redknot.cli.verify_vllm_sources",
                return_value={"status": "SOURCE_CONTRACT_OK"},
            ) as verify,
        ):
            result = doctor(str(self.config), "/fixture/vllm")
        verify.assert_called_once_with(
            "/fixture/vllm", engine_family="deepseek_v4_flash"
        )
        self.assertEqual(result["engine_family"], "deepseek_v4_flash")
        self.assertFalse(result["gpu_inference_verified"])
        self.assertFalse(result["gpu_initialized_by_doctor"])

    def test_flash_example_keeps_dense_boundaries_and_56_local_heads(self):
        from vllm_redknot.config import load_config

        example = Path(__file__).parents[1] / "examples/deepseek_v4_flash_policy.json"
        config = load_config(example)
        self.assertEqual(config["engine_family"], "deepseek_v4_flash")
        self.assertEqual(config["model_revision"], FLASH_REVISION)
        self.assertEqual(config["boundary_tokens"], 128)
        self.assertEqual(set(config["local_heads"]), {str(i) for i in range(3, 40)})
        expected = [head for head in range(64) if head % 8]
        self.assertEqual(len(expected), 56)
        for heads in config["local_heads"].values():
            self.assertEqual(heads, expected)
        self.assertEqual(config["max_cache_bytes"], 8 * 1024**3)
        raw = json.loads(example.read_text())
        self.assertNotIn("rotary_dim", raw)
        self.assertNotIn("rope_theta", raw)

    def test_template_identity_must_be_replaced(self):
        self.settings["model_revision"] = "REPLACE_WITH_REAL_REVISION"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Replace"):
            serve_command(str(self.model), str(self.config))

    def test_disabled_config_cannot_launch_reuse(self):
        self.settings["enabled"] = False
        self.write_config()
        with self.assertRaisesRegex(ValueError, "enabled"):
            serve_command(str(self.model), str(self.config))

    def test_launch_rejects_invalid_port_context_and_dtype(self):
        for options in ({"port": 0}, {"max_model_len": 1}, {"dtype": "fp8"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                serve_command(str(self.model), str(self.config), **options)

    def test_uninstalled_plugin_is_not_reported_installed(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            self.assertFalse(installed_plugin())


if __name__ == "__main__":
    unittest.main()
