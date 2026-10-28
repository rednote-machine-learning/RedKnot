"""Model and dataset inventory for the migrated RedKnot benchmark.

An entry here is a preparation profile, not a claim that DeepInfer implements
that model. Only the V4.1 profile can be submitted to the current native engine.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

SOURCE_COMMIT = "19c152a44eee0439afc85fe95ac23ce5a914c8fd"
SOURCE_REPOSITORY = "https://github.com/rednote-machine-learning/RedKnot"
SOURCE_ASSETS_SHA256 = "f8bd7d8e324f681cb9f96714326ff696048bc8a2f636a6cc990549e31801ac3a"
ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "sglang"
DATASETS = ("hotpotqa", "2wikimqa", "musique", "multifieldqa_en", "triviaqa")

SUITES = {
    "64k": {"name": "64k", "target_tokens": 65536, "num_chunks": 4, "chunk_tokens": 16384},
    "128k": {"name": "128k", "target_tokens": 131072, "num_chunks": 4, "chunk_tokens": 32768},
    "256k": {"name": "256k", "target_tokens": 262144, "num_chunks": 8, "chunk_tokens": 32768},
    "440k": {"name": "440k", "target_tokens": 450560, "num_chunks": 8, "chunk_tokens": 56320},
}
for _name, _suite in SUITES.items():
    _suite["path"] = f"datasets/LongBench/suites/release_{_name}_15case.jsonl"

MODEL_PROFILES = {
    "deepseek-v4.1-flash": {
        "model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "architecture": "DeepSeek V4.1 Flash MLA/MoE",
        "native_supported": True,
        "status": "native_baseline",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_DeepSeekV4Flash.py",
        "head_config": None,
        "ffn_config": None,
        "notes": "Native DeepInfer baseline. The V4-0731 RedKnot policy is not a validated V4.1 policy.",
    },
    "deepseek-v4-flash-0731": {
        "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "architecture": "DeepSeek V4 Flash MLA/MoE",
        "native_supported": False,
        "status": "configuration_only",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_DeepSeekV4Flash.py",
        "head_config": "head_class/deepseek_v4_flash_0731_redknot.json",
        "ffn_config": "sparse_ffn_params/deepseek_v4_flash_0731.json",
        "notes": "Original frozen SGLang release model; native DeepInfer execution is not supported.",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "architecture": "Mistral GQA/SWA",
        "native_supported": False,
        "status": "configuration_only",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py",
        "head_config": "head_class/mistral-7B_parity_98local_w3000.json",
        "ffn_config": "sparse_ffn_params/mistral-7B.json",
        "notes": "Original entrypoint uses Transformers and a SGLang RedKnot driver; native port pending.",
    },
    "qwen3-32b": {
        "model_id": "Qwen/Qwen3-32B",
        "architecture": "Qwen3 GQA",
        "native_supported": False,
        "status": "configuration_only",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_Qwen3_RAG.py",
        "head_config": "head_class/qwen3-32B_optimal_g15_lf_ret.json",
        "ffn_config": "sparse_ffn_params/qwen3-32B.json",
        "notes": "Source model-specific selection and policy are retained as provenance; native port pending.",
    },
    "qwen3.5-35b-a3b": {
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "architecture": "Qwen3.5 hybrid attention/MoE",
        "native_supported": False,
        "status": "configuration_only",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_Qwen35_397B_RAG.py",
        "head_config": "head_class/qwen3.5-35B-A3B_redknot.json",
        "ffn_config": "sparse_ffn_params/qwen3.5-35B-A3B.json",
        "notes": "Source script defaults to 35B-A3B despite its filename; 397B-A17B source policies are also archived.",
    },
    "llama-3.3-70b": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "architecture": "Llama3.3 GQA",
        "native_supported": False,
        "status": "configuration_only",
        "source_entrypoint": "test/srt/redknot/benchmark_RedKnot_Llama3.3_RAG.py",
        "head_config": "head_class/llama-70B_sweetspot_g10_w4096.json",
        "ffn_config": "sparse_ffn_params/llama-70B.json",
        "notes": "Experimental Transformers-based source benchmark; native port pending.",
    },
}
for _name, _profile in MODEL_PROFILES.items():
    _profile["profile"] = _name


def get_model_profile(name: str) -> dict:
    try:
        return deepcopy(MODEL_PROFILES[name])
    except KeyError as error:
        raise ValueError(f"unknown model profile {name!r}; choose from {', '.join(MODEL_PROFILES)}") from error


def get_suite(name: str) -> dict:
    normalized = name.lower()
    try:
        return deepcopy(SUITES[normalized])
    except KeyError as error:
        raise ValueError(f"unknown suite {name!r}; choose from {', '.join(SUITES)}") from error
