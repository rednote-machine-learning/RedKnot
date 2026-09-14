"""Explicit launch and CPU-only diagnostics for the opt-in extension."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from .compat import verify_vllm_sources
from .config import import_sglang_head_policy, load_config


def installed_plugin() -> bool:
    return any(
        entry.name == "redknot" and entry.value == "vllm_redknot.plugin:register"
        for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
    )


def checkpoint_engine_options(
    model_path: Path, config: dict[str, Any], dtype: str
) -> tuple[str, str]:
    """CPU-only family/geometry guard; return native backend and activation dtype.

    This does not authenticate checkpoint file contents or validate GPU support.
    The worker performs the full pinned model/runtime checks before execution.
    """
    checkpoint_path = model_path / "config.json"
    if not model_path.is_dir() or not checkpoint_path.is_file():
        raise ValueError("--model must name an existing local checkpoint directory")
    if dtype not in {"auto", "bfloat16", "float16"}:
        raise ValueError("Activation dtype must be auto, bfloat16 or float16")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint config.json must contain an object")
    family = config.get("engine_family", "mha")
    architectures = checkpoint.get("architectures", [])
    if family == "deepseek_v4_flash":
        from .dsv4_runner import FLASH_REVISION

        if architectures != ["DeepseekV4ForCausalLM"]:
            raise ValueError("deepseek_v4_flash requires DeepseekV4ForCausalLM")
        required = {
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "o_groups": 8,
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise ValueError("Only Flash-0731's 43-layer, 64-query-head MLA is adapted")
        if config["model_revision"] != FLASH_REVISION:
            raise ValueError(
                "deepseek_v4_flash requires the pinned Flash-0731 revision"
            )
        if checkpoint.get("vision_n_layers", 0):
            raise ValueError("Only text-only Flash-0731 is adapted")
        if dtype == "float16":
            raise ValueError(
                "Flash requires bfloat16 activations and native FP8 weights"
            )
        if any(
            head >= 64 for heads in config["local_heads"].values() for head in heads
        ):
            raise ValueError("Flash local heads are logical query-head IDs in [0, 64)")
        return "FLASHMLA_SPARSE_DSV4", "bfloat16"
    if family != "mha":
        raise ValueError(f"Unsupported engine_family: {family}")
    allowed = {"Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM"}
    if not architectures or any(name not in allowed for name in architectures):
        raise ValueError(
            "engine_family=mha supports Qwen2/Qwen3/Llama MHA/GQA only; "
            "Flash requires explicit engine_family=deepseek_v4_flash."
        )
    return "CUSTOM", dtype


def serve_command(
    model: str,
    config_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 18731,
    max_model_len: int = 32768,
    dtype: str = "bfloat16",
) -> tuple[list[str], dict[str, str]]:
    """Validate before creating a subprocess; constructing this starts no GPU."""
    model_path = Path(model).resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("--model must name an existing local checkpoint directory")
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    if config is None or not config["enabled"]:
        raise ValueError("serve requires an enabled, explicit configuration")
    revision = config["model_revision"].upper()
    if "REPLACE" in revision or "UNSET" in revision or "EXAMPLE" in revision:
        raise ValueError(
            "Replace the example model_revision with the checkpoint identity"
        )
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if type(max_model_len) is not int or max_model_len < 2:
        raise ValueError("max_model_len must be at least 2")
    backend, dtype = checkpoint_engine_options(model_path, config, dtype)
    environment = dict(os.environ)
    environment["VLLM_REDKNOT_CONFIG"] = str(config_file)
    environment["VLLM_PLUGINS"] = "redknot"
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # Never request model downloads as a side effect of this launch helper.
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_model_len),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--attention-config",
        json.dumps({"backend": backend}, separators=(",", ":")),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
    ]
    return command, environment


def doctor(config_path: str | None, vllm_root: str | None) -> dict[str, Any]:
    settings = load_config(config_path)
    family = settings.get("engine_family", "mha") if settings else "mha"
    result: dict[str, Any] = {
        "python": sys.executable,
        "plugin_installed": installed_plugin(),
        "enabled": bool(settings and settings["enabled"]),
        "gpu_initialized_by_doctor": False,
        "gpu_inference_verified": False,
        "engine_family": family,
        "deepseek_v4_serving_supported": family == "deepseek_v4_flash",
        "deepseek_v4_support_scope": (
            "experimental Flash-0731 TP1 projected local-head reuse; GPU unverified"
            if family == "deepseek_v4_flash"
            else None
        ),
    }
    try:
        result["source_contract"] = verify_vllm_sources(vllm_root, engine_family=family)
    except RuntimeError as error:
        result["source_contract"] = {"status": "FAILED", "reason": str(error)}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("code-map", help="Print RedKnot code markers; no model import")
    check = commands.add_parser("doctor", help="CPU-only source/config compatibility")
    check.add_argument("--config")
    check.add_argument("--vllm-root", help="Python vllm package directory (optional)")
    convert = commands.add_parser("import-head-policy", help="Print reviewed head IDs")
    convert.add_argument("input")
    convert.add_argument("--model-revision", required=True)
    convert.add_argument("--rope-theta", type=float, required=True)
    launch = commands.add_parser(
        "serve", help="Explicitly launch a GPU-serving process"
    )
    launch.add_argument("--model", required=True)
    launch.add_argument("--config", required=True)
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=18731)
    launch.add_argument("--max-model-len", type=int, default=32768)
    launch.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    launch.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "code-map":
            from .code_map import implementation_map

            print(json.dumps(implementation_map(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "doctor":
            result = doctor(args.config, args.vllm_root)
            print(json.dumps(result, indent=2))
            return (
                0 if result["source_contract"]["status"] == "SOURCE_CONTRACT_OK" else 2
            )
        if args.command == "import-head-policy":
            result = import_sglang_head_policy(
                json.loads(Path(args.input).read_text()),
                model_revision=args.model_revision,
                rope_theta=args.rope_theta,
            )
            print(json.dumps(result, indent=2))
            return 0
        command, environment = serve_command(
            args.model,
            args.config,
            host=args.host,
            port=args.port,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
        )
        family = load_config(args.config).get("engine_family", "mha")
        verify_vllm_sources(engine_family=family)
        if not installed_plugin():
            raise ValueError(
                "The worker plugin is not installed in this interpreter. "
                "Install this project with uv pip install --no-deps -e . "
                "in your separately prepared vLLM environment."
            )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "command": command,
                        "environment_overrides": {
                            key: environment[key]
                            for key in (
                                "VLLM_REDKNOT_CONFIG",
                                "VLLM_PLUGINS",
                                "VLLM_USE_V2_MODEL_RUNNER",
                                "HF_HUB_OFFLINE",
                                "TRANSFORMERS_OFFLINE",
                            )
                        },
                        "gpu_process_started": False,
                    },
                    indent=2,
                )
            )
            return 0
        os.execve(sys.executable, command, environment)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"vllm-redknot: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
