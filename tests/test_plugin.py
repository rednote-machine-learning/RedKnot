"""Engine-family registration is explicit and source checks precede mutations."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from vllm_redknot.config import validate_config
from vllm_redknot.plugin import register


class PluginTests(unittest.TestCase):
    def modules(self):
        modules = {}
        for name in (
            "vllm",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backends",
            "vllm.v1.attention.backends.registry",
            "vllm_redknot.runner",
            "vllm_redknot.dsv4_runner",
            "vllm_redknot.dsv4_backend",
        ):
            modules[name] = types.ModuleType(name)
        registry = modules["vllm.v1.attention.backends.registry"]
        registry.AttentionBackendEnum = types.SimpleNamespace(CUSTOM="custom")
        registry.register_backend = Mock()
        modules["vllm_redknot.runner"].install_runner_hooks = Mock()
        modules["vllm_redknot.dsv4_runner"].install_dsv4_runner = Mock()
        modules["vllm_redknot.dsv4_backend"].install_dsv4_attention = Mock()
        return modules

    def config(self, family="mha"):
        return validate_config(
            {
                "engine_family": family,
                "model_revision": "fixture",
                "local_heads": {"1": [0]},
            }
        )

    def test_unset_and_disabled_leave_all_framework_state_untouched(self):
        for config in (None, {"enabled": False}):
            with (
                patch("vllm_redknot.config.load_config", return_value=config),
                patch("vllm_redknot.compat.verify_vllm_sources") as verify,
            ):
                register()
                verify.assert_not_called()

    def test_mha_retains_custom_backend_registration(self):
        modules = self.modules()
        with (
            patch.dict(sys.modules, modules),
            patch("vllm_redknot.config.load_config", return_value=self.config()),
            patch("vllm_redknot.compat.verify_vllm_sources") as verify,
        ):
            register()
        verify.assert_called_once_with(engine_family="mha")
        modules[
            "vllm.v1.attention.backends.registry"
        ].register_backend.assert_called_once_with(
            "custom", "vllm_redknot.vllm_backend.RedKnotBackend"
        )
        modules["vllm_redknot.runner"].install_runner_hooks.assert_called_once()
        modules["vllm_redknot.dsv4_runner"].install_dsv4_runner.assert_not_called()
        modules["vllm_redknot.dsv4_backend"].install_dsv4_attention.assert_not_called()

    def test_dsv4_uses_native_hooks_without_mha_or_custom_backend(self):
        modules = self.modules()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "vllm_redknot.config.load_config",
                return_value=self.config("deepseek_v4_flash"),
            ),
            patch("vllm_redknot.compat.verify_vllm_sources") as verify,
        ):
            register()
        verify.assert_called_once_with(engine_family="deepseek_v4_flash")
        modules["vllm_redknot.dsv4_runner"].install_dsv4_runner.assert_called_once()
        modules[
            "vllm_redknot.dsv4_backend"
        ].install_dsv4_attention.assert_called_once_with()
        modules["vllm_redknot.runner"].install_runner_hooks.assert_not_called()
        modules[
            "vllm.v1.attention.backends.registry"
        ].register_backend.assert_not_called()

    def test_source_drift_prevents_all_registration_mutations(self):
        for family in ("mha", "deepseek_v4_flash"):
            modules = self.modules()
            with (
                patch.dict(sys.modules, modules),
                patch(
                    "vllm_redknot.config.load_config", return_value=self.config(family)
                ),
                patch(
                    "vllm_redknot.compat.verify_vllm_sources",
                    side_effect=RuntimeError("drift"),
                ),
                self.assertRaisesRegex(RuntimeError, "drift"),
            ):
                register()
            modules["vllm_redknot.runner"].install_runner_hooks.assert_not_called()
            modules["vllm_redknot.dsv4_runner"].install_dsv4_runner.assert_not_called()
            modules[
                "vllm_redknot.dsv4_backend"
            ].install_dsv4_attention.assert_not_called()
            modules[
                "vllm.v1.attention.backends.registry"
            ].register_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
