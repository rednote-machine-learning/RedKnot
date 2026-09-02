#!/usr/bin/env python3
"""Single entry point for DeepSeek-V4 native and RedKnot three-way tests.

The RedKnot mode intentionally combines only compatible mechanisms:

1. context-bound pure MLA offline-local / online-global head merge;
2. per-layer routed-expert progressive Top-K;
3. offline Indexer K/state with query-dependent Q, scoring and Top-512 online.

An explicit ``--row-sparse-online`` qualification arm reuses the repository's
existing checkpoint/Indexer replay to physically propagate only selected
document rows.  It is deliberately labelled separately from pure MLA-offload;
the arm is used to establish the systems ceiling before the two mechanisms are
combined, never to report a legacy fallback as pure head-split execution.

Token-drop sparse FFN stays disabled so accuracy loss and compute savings remain
attributable to the two approximation levers above.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DRIVER = REPO / "test/srt/redknot/benchmark_RedKnot_DeepSeekV4_Flash_RAG.py"
MODEL = Path(
    "/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/"
    "DeepSeek-V4-Flash-0731"
)
DATA_DIR = REPO / "test/srt/redknot/datasets/LongBench/data"
DEFAULT_DATA_MANIFEST = (
    REPO / "test/srt/redknot/musique_pure_prompt_selection_v1.json"
)
DATA_MANIFEST_128K = (
    REPO / "test/srt/redknot/musique_pure_prompt_selection_128k_v1.json"
)
DATA_MANIFEST_256K_32K = (
    REPO
    / "test/srt/redknot/musique_pure_prompt_selection_256k_32k_v1.json"
)

SELECTION_SHA256 = (
    "586fd683bfe043e1a6aaa1d07c7236ea9d956d99be739be743c4a2ec1728bcd8"
)
PROMPT_TEXT_SHA256 = (
    "sha256:fa33caccb16d22f9df544239de3229c74bf6ce6847148ddeccbdbde371db11c8"
)
FULL_INPUT_IDS_SHA256 = (
    "sha256:9329590a5c2bb87e7689d5d8b81edbadf50394a89f268df97268debd82bea891"
)
FULL_INPUT_TOKENS = 65585
OFFLINE_DOCUMENT_TOKENS = 65536

REDKNOT_TARGET_PROFILES = {
    65536: {
        "num_chunks": 8,
        "chunk_tokens": 8192,
        "query_row_id": 68,
        "data_manifest": DEFAULT_DATA_MANIFEST,
        "selection_sha256": SELECTION_SHA256,
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "full_input_ids_sha256": FULL_INPUT_IDS_SHA256,
        "full_input_tokens": FULL_INPUT_TOKENS,
        "mla_off_max_bytes": 8589934592,
        "mla_off_device_max_bytes": 5368709120,
    },
    131072: {
        "num_chunks": 16,
        "chunk_tokens": 8192,
        "query_row_id": 68,
        "data_manifest": DATA_MANIFEST_128K,
        "selection_sha256": (
            "caf99890880e0de190f845d0a38e600d760d2153cd1961888bd7776a2044f040"
        ),
        "prompt_text_sha256": (
            "sha256:9959bc0f32f7eb29a4cf61e7d7a20ca8fda937166057510f49ae74056576f4b1"
        ),
        "full_input_ids_sha256": (
            "sha256:3b1ee37110db315a9ba84a3ae55adfce61b2aaa61520fc4c68511313cf96dd87"
        ),
        "full_input_tokens": 131128,
        # 16 context-bound 8K artifacts project to 9,938,141,184 bytes/rank.
        "mla_off_max_bytes": 12884901888,
        "mla_off_device_max_bytes": 10737418240,
    },
    262144: {
        "num_chunks": 8,
        "chunk_tokens": 32768,
        "query_row_id": 0,
        "data_manifest": DATA_MANIFEST_256K_32K,
        "selection_sha256": (
            "a2524b87a6ff0a91e7f5aef104d3b8eb14b9aa55e2b8c6b5db34ef0dbe1477cc"
        ),
        # Byte-frozen official one-pass prompt identity.  These values were
        # produced before any GPU/model request from the immutable data
        # selection above.
        "prompt_text_sha256": (
            "sha256:c2bb701688ef4cffc4911cf325c1928c2517f2270ea24e84cd9f492877ae6b4e"
        ),
        "full_input_ids_sha256": (
            "sha256:6adc0143211e3c7d4593ceee923dbeaa923e6f53c7ab02c4f42071ff2dffd310"
        ),
        "full_input_tokens": 262197,
        # 8 context-bound 32K artifacts project to about 19.9 GiB/rank.
        "mla_off_max_bytes": 25769803776,
        "mla_off_device_max_bytes": 22548578304,
    },
    450560: {
        "num_chunks": 8,
        "chunk_tokens": 56320,
        # 440K (binary) keeps the eight-document RAG geometry while avoiding
        # the eighth 64K snapshot's measured 4 GiB publication OOM.
        "query_row_id": 0,
        "data_manifest": Path(""),
        "selection_sha256": "",
        "prompt_text_sha256": "",
        "full_input_ids_sha256": "",
        "full_input_tokens": 0,
        # Scale the certified 512K CPU-authoritative artifact budget by 55/64.
        "mla_off_max_bytes": 44291850240,
        "mla_off_device_max_bytes": 0,
        "requires_qualification_profile": True,
    },
    524288: {
        "num_chunks": 8,
        "chunk_tokens": 65536,
        # 512K is always driven by an immutable multi-dataset qualification
        # profile.  There is deliberately no mutable/default corpus identity.
        "query_row_id": 0,
        "data_manifest": Path(""),
        "selection_sha256": "",
        "prompt_text_sha256": "",
        "full_input_ids_sha256": "",
        "full_input_tokens": 0,
        # The exact preflight projection is 39,752,564,736 bytes/rank for
        # 8x64K.  Keep explicit CPU/device headroom without hiding the cost.
        "mla_off_max_bytes": 51539607552,
        # A complete 8x64K BF16 z_off bank is ~37 GiB per TP rank.  Keeping
        # that bank resident together with the >=512K scheduler KV pool leaves
        # less than one 64K-prefill temporary on a 140 GiB B300 and OOMs while
        # publishing the fourth segment.  The controller already has a
        # fail-closed CPU-authoritative restore path: assemble one layer's
        # clean rows on CPU, transfer that layer once, then release it before
        # advancing.  Use that bounded-memory path for 512K; the follow-up
        # optimization is a two-layer pinned prefetch ring, not an impossible
        # all-segment device mirror.
        "mla_off_device_max_bytes": 0,
        "requires_qualification_profile": True,
    },
}


def _resolve_redknot_target_profile(args) -> dict:
    base = REDKNOT_TARGET_PROFILES.get(args.target_tokens)
    if base is None:
        raise ValueError(
            "context-bound RedKnot currently supports exactly 65536, "
            "131072, 262144, 450560, or 524288 offline document tokens"
        )
    profile = dict(base)
    profile["dataset"] = "musique"
    profile["num_queries"] = 1
    profile["query_row_ids"] = [int(profile["query_row_id"])]
    profile["prompt_manifest"] = ""
    profile["prompt_manifest_sha256"] = ""
    if not args.qualification_profile:
        if profile.get("requires_qualification_profile"):
            raise ValueError(
                "512K requires an immutable multi-dataset qualification profile"
            )
        return profile
    source = Path(args.qualification_profile).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    profile_format = document.get("format")
    if profile_format not in {
        "redknot_multidataset_256k_profile_v1",
        "redknot_multidataset_profile_v2",
    }:
        raise ValueError("unsupported multi-dataset qualification profile")
    if profile_format == "redknot_multidataset_256k_profile_v1" and (
        args.target_tokens != 262144
    ):
        raise ValueError("v1 multi-dataset profiles are frozen at 256K")
    if profile_format == "redknot_multidataset_profile_v2" and (
        document.get("target_tokens") != args.target_tokens
    ):
        raise ValueError(
            "v2 qualification profile target length differs from the request"
        )
    for key, expected in (
        ("num_chunks", int(base["num_chunks"])),
        ("chunk_tokens", int(base["chunk_tokens"])),
        ("query_start", int(args.target_tokens)),
    ):
        if document.get(key) != expected:
            raise ValueError(
                f"qualification profile has invalid {key}: {document.get(key)!r}"
            )
    query_rows = document.get("query_row_ids")
    num_queries = document.get("num_queries")
    if (
        type(num_queries) is not int
        or num_queries <= 0
        or not isinstance(query_rows, list)
        or len(query_rows) != num_queries
        or any(type(value) is not int or value < 0 for value in query_rows)
        or len(set(query_rows)) != len(query_rows)
    ):
        raise ValueError("qualification profile has invalid query rows")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != num_queries:
        raise ValueError("qualification profile has invalid prompt cases")
    profile.update(
        {
            "dataset": str(document["dataset"]),
            "num_queries": num_queries,
            "query_row_id": query_rows[0],
            "query_row_ids": query_rows,
            "data_manifest": Path(document["data_manifest"]),
            "selection_sha256": str(document["data_selection_sha256"]),
            "prompt_manifest": str(document["prompt_manifest"]),
            "prompt_manifest_sha256": str(
                document["prompt_manifest_sha256"]
            ),
            "prompt_text_sha256": str(cases[0]["text_sha256"]),
            "full_input_ids_sha256": str(
                cases[0]["full_input_ids_sha256"]
            ),
            "full_input_tokens": int(document["max_total_tokens"]),
        }
    )
    profile.pop("requires_qualification_profile", None)
    return profile


def _set(name: str, value: object) -> None:
    os.environ[name] = str(value)


def _strict_binary_env(name: str) -> str:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be exactly 0 or 1, got {value!r}")
    return value


def _validate_system_optimizer_dependencies() -> None:
    if os.environ.get("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0") == "1":
        import deep_gemm

        if not callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
            raise RuntimeError(
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 requires "
                "deep_gemm.tf32_hc_prenorm_gemm; this runtime does not "
                "provide that kernel"
            )


def _unset_legacy_environment() -> None:
    for name in (
        "REDKNOT_HEAD_CFG",
        "REDKNOT_C4_TOPK_CLAMP",
        "REDKNOT_IH_DATA_MANIFEST_OUT",
        "REDKNOT_IH_DATA_EXCLUDE_MANIFESTS",
        "REDKNOT_IH_SELECTION_POLICY",
        "REDKNOT_IH_CHECKPOINT_STRIDE",
        "REDKNOT_IH_CHECKPOINT_MAX_ISLANDS",
        "REDKNOT_IH_ACTIVE_BUDGET_RATIO",
        "REDKNOT_IH_QUERY_PROTECTION_TOKENS",
        "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER",
        "REDKNOT_IH_GENERALIZED_STRONG_ACTIVE_RATIO",
        "REDKNOT_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO",
        "REDKNOT_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO",
        "REDKNOT_IH_HOT_MAX_PER_SEGMENT_RATIO",
        "REDKNOT_IH_HOT_FRAC",
        "REDKNOT_IH_SKIP_PREFIX_RECOMPUTE",
        "REDKNOT_IH_SERVER_POLICY_MANIFEST",
        "REDKNOT_IH_SERVER_INSTANCE_NONCE",
        "REDKNOT_IH_PROMPT_MANIFEST",
    ):
        os.environ.pop(name, None)


def _prepare_output_paths(args) -> tuple[Path, Path, Path, Path]:
    result = Path(args.out).expanduser().resolve()
    server_log = Path(args.log).expanduser().resolve()
    target_profile = (
        _resolve_redknot_target_profile(args) if args.mode == "redknot" else None
    )
    prompt_manifest_input = (
        Path(target_profile["prompt_manifest"]).expanduser().resolve()
        if target_profile and target_profile.get("prompt_manifest")
        else None
    )
    prompt_manifest = prompt_manifest_input or (
        Path(args.prompt_manifest_out).expanduser().resolve()
        if args.prompt_manifest_out
        else result.with_name(result.stem + ".prompt_manifest.json")
    )
    rank_logs = (
        Path(args.rank_log_dir).expanduser().resolve()
        if args.rank_log_dir
        else result.with_name(result.stem + ".ranklogs")
    )
    for path in (result, server_log):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite benchmark artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_manifest_input is not None:
        if not prompt_manifest_input.is_file():
            raise FileNotFoundError(
                f"frozen prompt manifest is absent: {prompt_manifest_input}"
            )
    else:
        if prompt_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite benchmark artifact: {prompt_manifest}"
            )
        prompt_manifest.parent.mkdir(parents=True, exist_ok=True)
    rank_logs.mkdir(parents=True, exist_ok=False)
    return result, server_log, prompt_manifest, rank_logs


def _configure_native(args, result: Path) -> None:
    length_label = {
        8000: "8K",
        16000: "16K",
        32000: "32K",
        64000: "64K",
        128000: "128K",
        256000: "256K",
    }.get(args.target_tokens)
    if length_label is None:
        raise ValueError("native target tokens must be 8K/16K/32K/64K/128K/256K")
    _set("REDKNOT_ENGINE_MODE", "baseline")
    _set("REDKNOT_DATASETS", args.datasets)
    _set("REDKNOT_LENGTHS", length_label)
    _set("REDKNOT_N_SAMPLES", args.n_samples)
    _set("REDKNOT_MAX_NEW", args.max_new)
    _set("REDKNOT_RESULT_OUT", result)
    _set("REDKNOT_SPARSE_FFN", 0)
    _set("REDKNOT_THREE_WAY_CLOSURE", 0)
    _set("REDKNOT_PROGRESSIVE_TOPK_SCHEDULE", "")
    _set("REDKNOT_ADAPTIVE_TOPK", int(args.adaptive_topk))
    _set("REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED", 0)
    _set("REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS", args.adaptive_topk_mass)
    _set("REDKNOT_ADAPTIVE_TOPK_BUCKETS", args.adaptive_topk_buckets)
    _set("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS", args.adaptive_topk_min_tokens)
    _set("REDKNOT_ENGINE_WARMUP_ITERS", args.engine_warmup_iters)
    _set(
        "REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM",
        int(args.adaptive_topk_log_histogram),
    )


def _configure_redknot(
    args,
    result: Path,
    server_log: Path,
    prompt_manifest: Path,
    rank_logs: Path,
) -> None:
    combined_headsplit_row_sparse = bool(args.combined_headsplit_row_sparse)
    row_sparse_enabled = bool(
        args.row_sparse_online or combined_headsplit_row_sparse
    )
    if args.adaptive_topk and args.progressive_topk_schedule:
        raise ValueError(
            "adaptive Top-K and progressive Top-K are mutually exclusive; "
            "use --progressive-topk-schedule '' for the physical adaptive run"
        )
    if row_sparse_enabled and args.prefix_materialization:
        raise ValueError(
            "row-sparse qualification cannot use full radix materialization"
        )
    if row_sparse_enabled and not args.first_document_prefix:
        raise ValueError(
            "row-sparse qualification requires --first-document-prefix; "
            "document 1 is served entirely from its offline artifact"
        )
    # A combined selected-row request is allowed to consume exactly one
    # scheduler-authenticated document-1 radix prefix.  The runtime binds the
    # seed and consumer to the same token hash, receipt key and compressor
    # terminal-state receipt; documents 2..N remain on the combined
    # head-split + selected-row path.  Full-prefix materialization stays
    # forbidden above for row-sparse qualification.
    radix_prefix_enabled = bool(
        args.prefix_materialization or args.first_document_prefix
    )
    if radix_prefix_enabled and args.radix_eviction_policy != "lfu":
        raise ValueError(
            "prefix materialization requires radix eviction policy=lfu so "
            "the repeatedly reused fixed prefix is not displaced by one-shot "
            "dense namespaces under concurrent load"
        )
    target_profile = _resolve_redknot_target_profile(args)
    if (
        not target_profile["prompt_text_sha256"]
        or not target_profile["full_input_ids_sha256"]
        or int(target_profile["full_input_tokens"]) <= 0
    ):
        raise ValueError(
            "the selected target profile is missing its frozen official-prompt "
            "identity"
        )
    data_manifest = Path(
        args.data_manifest or target_profile["data_manifest"]
    ).expanduser().resolve()
    dataset_name = str(target_profile["dataset"])
    for path in (
        DRIVER,
        MODEL / "config.json",
        DATA_DIR / f"{dataset_name}.jsonl",
        data_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required benchmark input is absent: {path}")

    _unset_legacy_environment()
    _set("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    _set("REDKNOT_MODEL_PATH", MODEL)
    _set("REDKNOT_LONGBENCH_DIR", DATA_DIR)
    _set("REDKNOT_DATASETS", dataset_name)
    _set("REDKNOT_ENGINE_MODE", "indexer_hot")
    _set("REDKNOT_IH_NO_LAUNCH", 0)

    # Pure context-bound MLA head split, or the separately-labelled
    # selected-row systems-ceiling arm.  The latter intentionally keeps
    # MLA-off disabled until its accuracy/speed contract is established.
    _set(
        "REDKNOT_IH_MLA_OFFLOAD",
        int(combined_headsplit_row_sparse or not row_sparse_enabled),
    )
    _set("REDKNOT_IH_ROW_SPARSE_CLOSURE", int(row_sparse_enabled))
    _set(
        "REDKNOT_IH_COMBINED_HEADSPLIT_ROW_SPARSE",
        int(combined_headsplit_row_sparse),
    )
    _set("REDKNOT_IH_TP_SIZE", 8)
    _set("REDKNOT_TP_SIZE", 8)
    _set("REDKNOT_V4_MODE", "aggressive")
    _set("REDKNOT_MLA_PASS_MODE", "headwise")
    _set("REDKNOT_MLA_DENSE_PREFIX_LAYERS", 3)
    _set("REDKNOT_MLA_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_MLA_LOCAL_WINDOW", 128)
    _set("REDKNOT_MLA_GLOBAL_HEAD_STRIDE", 8)
    _set("REDKNOT_MLA_GLOBAL_LAYER_STRIDE", 0)
    _set("REDKNOT_MLA_REUSE_HEADS_FULL_SCOPE", 1)
    _set("REDKNOT_SHARED_LATENT_GPU", 1)
    # The current sparse-FFN selector is certified only for an exact pure-MLA
    # restore receipt.  Do not weaken that contract for the standalone
    # selected-row systems-ceiling arm: physical row pruning already removes
    # its skipped tokens, while adaptive routed K remains active.
    sparse_ffn_enabled = bool(
        args.token_sparse_ffn
        and (combined_headsplit_row_sparse or not row_sparse_enabled)
    )
    _set("REDKNOT_SPARSE_FFN", int(sparse_ffn_enabled))
    _set("REDKNOT_THREE_WAY_CLOSURE", int(sparse_ffn_enabled))
    # start_server_redknot_mla.sh intentionally does not forward sparse-FFN
    # CLI knobs when the feature is disabled.  Normalize inactive policy
    # metadata to the server defaults as well; otherwise benchmark preflight
    # hashes user-supplied-but-inactive values while the runtime manifest
    # hashes its defaults and the pure MLA run fails before the first snapshot.
    sparse_mass = args.token_sparse_mass if sparse_ffn_enabled else 0.60
    sparse_deep_mass = args.token_sparse_deep_mass if sparse_ffn_enabled else 0.60
    sparse_importance = (
        args.token_sparse_importance if sparse_ffn_enabled else "activation"
    )
    sparse_min_full_ratio = (
        args.token_sparse_min_full_ratio if sparse_ffn_enabled else 0.20
    )
    sparse_max_full_ratio = (
        args.token_sparse_max_full_ratio if sparse_ffn_enabled else 0.80
    )
    _set("REDKNOT_FFN_DENSE_UNTIL", 3)
    _set("REDKNOT_FFN_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_FFN_MASS", sparse_mass)
    _set("REDKNOT_FFN_DEEP_START", args.token_sparse_deep_start)
    _set("REDKNOT_FFN_MASS_DEEP", sparse_deep_mass)
    _set("REDKNOT_FFN_RECENT_N", args.token_sparse_recent_tokens)
    _set("REDKNOT_FFN_BOUNDARY_TOKENS", args.token_sparse_boundary_tokens)
    _set("REDKNOT_FFN_MIN_SEQ_LEN", args.token_sparse_min_seq_len)
    _set("REDKNOT_FFN_IMPORTANCE", sparse_importance)
    _set("REDKNOT_FFN_MIN_FULL_RATIO", sparse_min_full_ratio)
    _set("REDKNOT_FFN_MAX_FULL_RATIO", sparse_max_full_ratio)
    _set("REDKNOT_FFN_BLOCK_TOKENS", args.token_sparse_block_tokens)
    _set(
        "REDKNOT_FFN_FREEZE_BLOCK_SELECTION",
        int(args.token_sparse_freeze_blocks),
    )
    _set("SGLANG_REDKNOT_FFN_DEBUG", int(args.token_sparse_ffn))
    _set(
        "REDKNOT_IH_PREFIX_MATERIALIZATION",
        int(radix_prefix_enabled),
    )
    _set(
        "REDKNOT_IH_PREFIX_MATERIALIZATION_SCOPE",
        (
            "first_document"
            if args.first_document_prefix
            else ("full" if args.prefix_materialization else "none")
        ),
    )
    if row_sparse_enabled:
        _set("REDKNOT_IH_SELECTION_POLICY", "checkpoint_islands")
        _set("REDKNOT_IH_CHECKPOINT_STRIDE", args.row_sparse_checkpoint_stride)
        _set(
            "REDKNOT_IH_CHECKPOINT_MAX_ISLANDS",
            args.row_sparse_checkpoint_max_islands,
        )
        _set(
            "REDKNOT_IH_ACTIVE_BUDGET_RATIO",
            args.row_sparse_active_ratio,
        )
        _set(
            "REDKNOT_IH_QUERY_PROTECTION_TOKENS",
            args.query_protection_tokens,
        )
        _set(
            "REDKNOT_IH_HOT_MAX_PER_SEGMENT_RATIO",
            args.row_sparse_segment_cap_ratio,
        )
        _set("REDKNOT_IH_SKIP_PREFIX_RECOMPUTE", 1)
        _set(
            "REDKNOT_IH_GENERALIZED_ADAPTIVE_CONTROLLER",
            int(args.generalized_adaptive_controller),
        )
        _set(
            "REDKNOT_IH_GENERALIZED_STRONG_ACTIVE_RATIO",
            args.generalized_strong_active_ratio,
        )
        _set(
            "REDKNOT_IH_GENERALIZED_MEDIUM_ACTIVE_RATIO",
            args.generalized_medium_active_ratio,
        )
        _set(
            "REDKNOT_IH_GENERALIZED_DIFFUSE_ACTIVE_RATIO",
            args.generalized_diffuse_active_ratio,
        )
    _set("REDKNOT_RADIX_EVICTION_POLICY", args.radix_eviction_policy)
    _set(
        "REDKNOT_MLA_OFF_GEOMETRY_TEMPLATE_CACHE",
        int(args.geometry_template_cache),
    )
    _set(
        "REDKNOT_IH_MLA_OFF_RESTORE_PIPELINE_GROUP_LAYERS",
        args.restore_pipeline_group_layers,
    )

    # Assignment-sparse MoE. Every token is preserved; only routed K changes.
    _set("REDKNOT_PROGRESSIVE_TOPK_SCHEDULE", args.progressive_topk_schedule)
    _set("REDKNOT_ADAPTIVE_TOPK", int(args.adaptive_topk))
    _set(
        "REDKNOT_ADAPTIVE_TOPK_PLAN_SCOPED",
        int(combined_headsplit_row_sparse),
    )
    _set("REDKNOT_ADAPTIVE_TOPK_CUMULATIVE_MASS", args.adaptive_topk_mass)
    _set("REDKNOT_ADAPTIVE_TOPK_BUCKETS", args.adaptive_topk_buckets)
    _set("REDKNOT_ADAPTIVE_TOPK_MIN_TOKENS", args.adaptive_topk_min_tokens)
    _set("REDKNOT_ADAPTIVE_TOPK_DENSE_PREFIX_LAYERS", 3)
    _set("REDKNOT_ADAPTIVE_TOPK_DENSE_SUFFIX_LAYERS", 3)
    _set("REDKNOT_ADAPTIVE_TOPK_PHYSICAL_COMPACTION", 1)
    _set(
        "REDKNOT_ADAPTIVE_TOPK_LOG_FIRST_HISTOGRAM",
        int(args.adaptive_topk_log_histogram),
    )

    _set("REDKNOT_IH_NUM_CHUNKS", target_profile["num_chunks"])
    _set("REDKNOT_IH_CHUNK_TOKENS", target_profile["chunk_tokens"])
    if args.native_indexer_doc_cap:
        if int(target_profile["num_chunks"]) != int(args.native_indexer_documents):
            raise ValueError(
                "native Indexer document count must equal the frozen offline "
                "chunk count"
            )
        if int(target_profile["chunk_tokens"]) % 4:
            raise ValueError("native Indexer C4 document geometry is not integral")
        _set("REDKNOT_NATIVE_INDEXER_DOC_CAP", args.native_indexer_doc_cap)
        _set(
            "REDKNOT_NATIVE_INDEXER_DOCUMENTS",
            args.native_indexer_documents,
        )
        _set(
            "REDKNOT_NATIVE_INDEXER_C4_ROWS_PER_DOCUMENT",
            int(target_profile["chunk_tokens"]) // 4,
        )
    else:
        for name in (
            "REDKNOT_NATIVE_INDEXER_DOC_CAP",
            "REDKNOT_NATIVE_INDEXER_DOCUMENTS",
            "REDKNOT_NATIVE_INDEXER_C4_ROWS_PER_DOCUMENT",
        ):
            os.environ.pop(name, None)
    # Pure independent-document artifacts are captured at local position zero
    # and relocated online.  Every nonzero destination segment therefore
    # recomputes its first 128 local-head rows; context-bound boundary=0 is a
    # different algorithm and must never be selected by this entry point.
    _set("REDKNOT_IH_BOUNDARY", 128)
    _set("REDKNOT_IH_MERGED_PREFILL_TOKENS", args.merged_prefill_tokens)
    _set("REDKNOT_IH_MLA_OFF_REFRESH_LAYER_STRIDE", 0)
    _set("REDKNOT_IH_MLA_OFF_HOT_EXPAND_TOKENS", 0)
    _set("REDKNOT_IH_MLA_OFF_COMPACT_WOA", 0)
    _set(
        "REDKNOT_IH_MLA_OFF_DIAGNOSTIC_ABLATION",
        "zoff_only" if combined_headsplit_row_sparse else "full",
    )
    _set(
        "REDKNOT_IH_MLA_OFF_CERTIFIED_MAX_CONTEXT_TOKENS",
        target_profile["full_input_tokens"]
        if combined_headsplit_row_sparse
        else 0,
    )
    _set(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_ONLY",
        int(not row_sparse_enabled),
    )
    _set(
        "REDKNOT_IH_MLA_OFF_QUALIFICATION_MAX_CONTEXT_TOKENS",
        0 if row_sparse_enabled else target_profile["full_input_tokens"],
    )

    # Frozen real MuSiQue row and official DSV4 prompt.
    _set("REDKNOT_IH_DATA_ROW_OFFSET", 0)
    _set("REDKNOT_IH_DATA_MANIFEST", data_manifest)
    _set(
        "REDKNOT_IH_EXPECTED_DATA_SELECTION_SHA256",
        target_profile["selection_sha256"],
    )
    _set("REDKNOT_IH_EXPECTED_DATASET", dataset_name)
    _set("REDKNOT_IH_EXPECTED_QUERY_ROW_ID", target_profile["query_row_id"])
    _set(
        "REDKNOT_IH_EXPECTED_QUERY_ROW_IDS",
        ",".join(map(str, target_profile["query_row_ids"])),
    )
    _set("REDKNOT_IH_PURE_PROMPT_MODE", "official_rag_v1")
    _set("REDKNOT_THINKING_MODE", "chat")
    _set("REDKNOT_REASONING_EFFORT", "low")
    _set(
        "REDKNOT_IH_EXPECTED_PROMPT_TEXT_SHA256",
        target_profile["prompt_text_sha256"],
    )
    _set(
        "REDKNOT_IH_EXPECTED_FULL_INPUT_IDS_SHA256",
        target_profile["full_input_ids_sha256"],
    )
    _set(
        "REDKNOT_IH_EXPECTED_FULL_INPUT_TOKENS",
        target_profile["full_input_tokens"],
    )
    if target_profile.get("prompt_manifest"):
        _set("REDKNOT_IH_PROMPT_MANIFEST", prompt_manifest)
        os.environ.pop("REDKNOT_IH_PROMPT_MANIFEST_OUT", None)
    else:
        _set("REDKNOT_IH_PROMPT_MANIFEST_OUT", prompt_manifest)
        os.environ.pop("REDKNOT_IH_PROMPT_MANIFEST", None)
    _set(
        "REDKNOT_IH_EXPECTED_PROMPT_MANIFEST_SHA256",
        target_profile.get("prompt_manifest_sha256", ""),
    )
    _set("REDKNOT_IH_NUM_QUERIES", target_profile["num_queries"])
    _set("REDKNOT_IH_QUALITY_REPEATS", args.quality_repeats)
    _set("REDKNOT_IH_MAX_NEW", args.max_new)
    _set("REDKNOT_IH_RELEVANCE_FIRST", 0)
    _set("REDKNOT_IH_RELEVANCE_LAST", 0)

    # Accuracy first; timing is diagnostic until fidelity passes.
    _set("REDKNOT_IH_TTFT_WARMUP", args.ttft_warmup)
    _set("REDKNOT_IH_TTFT_ITERS", args.ttft_iters)
    _set("REDKNOT_IH_MEASURE_QPS", int(args.measure_qps))
    _set("REDKNOT_IH_QPS_CONCURRENCIES", args.qps_concurrencies)
    _set("REDKNOT_IH_QPS_WARMUP_WAVES", args.qps_warmup_waves)
    _set("REDKNOT_IH_QPS_WAVES", args.qps_waves)
    _set("REDKNOT_IH_STRICT_PERFORMANCE_CLAIMS", int(args.strict_performance))
    _set("REDKNOT_IH_MIN_TOP1_RATE", 0.95)
    _set("REDKNOT_IH_MIN_COSINE", args.min_cosine)
    _set("REDKNOT_IH_MIN_F1_RETENTION", 0.98)
    _set("REDKNOT_IH_MIN_EM_RETENTION", 0.98)
    _set("REDKNOT_IH_MIN_DENSE_F1", 0.0)
    _set("REDKNOT_IH_MIN_REUSE_F1", 0.0)
    _set("REDKNOT_IH_MIN_DENSE_EM", 0.0)
    _set("REDKNOT_IH_MIN_REUSE_EM", 0.0)
    _set("REDKNOT_IH_MIN_TOKEN_AGREEMENT", args.min_token_agreement)
    _set("REDKNOT_IH_MIN_SPEEDUP", args.ttft_target_speedup)
    _set("REDKNOT_IH_MIN_QPS_SPEEDUP", 1.5 if row_sparse_enabled else 2.0)
    _set("REDKNOT_IH_MIN_HEAD_ROW_SAVING", 0.70)

    _set("REDKNOT_IH_MLA_OFF_MAX_BYTES", target_profile["mla_off_max_bytes"])
    _set(
        "REDKNOT_MLA_OFF_DEVICE_MAX_BYTES",
        target_profile["mla_off_device_max_bytes"],
    )
    _set("REDKNOT_IH_MIN_GPU_FREE_MIB", args.min_gpu_free_mib)
    _set("REDKNOT_MEM_FRACTION_STATIC", args.mem_fraction_static)
    _set("REDKNOT_MOE_RUNNER_BACKEND", "marlin")
    _set("REDKNOT_DISABLE_CUDA_GRAPH", 1)
    _set("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", 1)
    _set("SGLANG_BARE_SUBPROCESS_LAUNCH", 1)
    # System-only kernel switches are explicit, binary and preserved from the
    # launcher.  Their exact values are also emitted into the structured
    # benchmark result, so a faster run cannot be mistaken for the default
    # implementation or silently inherit a malformed environment value.
    for name in (
        "SGLANG_OPT_USE_TILELANG_MHC_PRE",
        "SGLANG_OPT_USE_TILELANG_MHC_POST",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
        "REDKNOT_MLA_OFF_CUBLAS_WOA_FASTPATH",
    ):
        _set(name, _strict_binary_env(name))
    _validate_system_optimizer_dependencies()
    _set("SGLANG_OPT_USE_TOPK_V2", 0)
    # Torch renamed this variable, while the pinned cluster build still accepts
    # the legacy CUDA spelling.  Set both to the identical value so the 256K
    # snapshot tail cannot silently run with the default fragmented allocator.
    _set("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    _set("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _set("PYTHONUNBUFFERED", 1)
    _set("NO_PROXY", "127.0.0.1,localhost")
    _set("REDKNOT_IH_PORT", args.port)
    _set("REDKNOT_IH_VENV_PY", REPO / ".venv_tf5/bin/python")
    _set("REDKNOT_IH_SERVER_SCRIPT", REPO / "server/start_server_redknot_mla.sh")
    _set("REDKNOT_IH_SERVER_LOG", server_log)
    _set("REDKNOT_IH_RANK_LOG_DIR", rank_logs)
    _set("REDKNOT_ENABLE_METRICS", 1)
    _set("REDKNOT_IH_REQUIRE_MODEL_TTFT", 1)
    _set("REDKNOT_RESULT_OUT", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "redknot"), required=True)
    parser.add_argument("--port", type=int, default=31998)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--datasets",
        default="musique,hotpotqa,2wikimqa,multifieldqa_en",
        help="Comma-separated real LongBench datasets for baseline/full-recompute mode.",
    )
    parser.add_argument("--target-tokens", type=int, default=OFFLINE_DOCUMENT_TOKENS)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument(
        "--data-manifest",
        default="",
        help="Optional exact data manifest override; default follows target length.",
    )
    parser.add_argument(
        "--qualification-profile",
        default="",
        help=(
            "Frozen multi-query 256K dataset/prompt profile. RedKnot replays "
            "its exact data and prompt manifests before server launch."
        ),
    )
    parser.add_argument("--prompt-manifest-out", default="")
    parser.add_argument("--rank-log-dir", default="")
    parser.add_argument(
        "--progressive-topk-schedule",
        default="0-11:6,12-27:5,28-42:4",
        help="Inclusive layer schedule; empty string keeps native K=6.",
    )
    parser.add_argument("--adaptive-topk", action="store_true")
    parser.add_argument("--adaptive-topk-mass", type=float, default=0.90)
    parser.add_argument("--adaptive-topk-buckets", default="4,5,6")
    parser.add_argument("--adaptive-topk-min-tokens", type=int, default=512)
    parser.add_argument("--adaptive-topk-log-histogram", action="store_true")
    parser.add_argument(
        "--token-sparse-ffn",
        action="store_true",
        help=(
            "Qualification-only third closure lever for certified pure-MLA "
            "restore, including independent position-0/RoPE relocation: "
            "shared expert on every token and compact routed experts only on "
            "protected/high-salience tokens."
        ),
    )
    parser.add_argument("--token-sparse-mass", type=float, default=0.95)
    parser.add_argument("--token-sparse-deep-start", type=int, default=24)
    parser.add_argument("--token-sparse-deep-mass", type=float, default=0.90)
    parser.add_argument("--token-sparse-recent-tokens", type=int, default=256)
    parser.add_argument("--token-sparse-boundary-tokens", type=int, default=128)
    parser.add_argument("--token-sparse-min-seq-len", type=int, default=32768)
    parser.add_argument(
        "--token-sparse-importance",
        choices=("activation", "blend", "indexer", "indexer_indegree"),
        default="blend",
    )
    parser.add_argument("--token-sparse-min-full-ratio", type=float, default=0.10)
    parser.add_argument("--token-sparse-max-full-ratio", type=float, default=0.50)
    parser.add_argument(
        "--token-sparse-block-tokens",
        type=int,
        default=0,
        help=(
            "Select complete absolute-position blocks for routed MoE; 0 keeps "
            "the historical individual-token selector and 128 enables the "
            "RedKnot block-sparse systems path."
        ),
    )
    parser.add_argument(
        "--token-sparse-freeze-blocks",
        action="store_true",
        help=(
            "Freeze the first eligible layer's block selection for the rest "
            "of the current prefill microforward."
        ),
    )
    parser.add_argument(
        "--engine-warmup-iters",
        type=int,
        default=0,
        help="Unmeasured same-prompt engine warmups before baseline/full-recompute metrics.",
    )
    parser.add_argument("--quality-repeats", type=int, default=1)
    parser.add_argument(
        "--prefix-materialization",
        action="store_true",
        help=(
            "Explicit RedKnot context-bound prefix materialization: retain the "
            "hash-bound offline prefix in the device radix tree and execute "
            "only the online suffix. This is audited separately from ordinary "
            "pure head-row restore."
        ),
    )
    parser.add_argument(
        "--first-document-prefix",
        action="store_true",
        help=(
            "Seed only document 1 into radix cache; every online request "
            "must then execute certified pure RedKnot restore/merge for "
            "documents 2-8 and the query suffix."
        ),
    )
    parser.add_argument(
        "--row-sparse-online",
        action="store_true",
        help=(
            "Explicit systems-ceiling arm: restore offline C4/C128/Indexer "
            "state and propagate only checkpoint-selected document rows plus "
            "the query. This is not labelled as pure MLA head-split."
        ),
    )
    parser.add_argument(
        "--combined-headsplit-row-sparse",
        action="store_true",
        help=(
            "Experimental closed loop: independent position-0 local-head "
            "z_off artifacts + online global/dirty-local head merge, while "
            "Indexer checkpoint islands physically prune transformer rows."
        ),
    )
    parser.add_argument("--row-sparse-active-ratio", type=float, default=0.20)
    parser.add_argument(
        "--generalized-adaptive-controller",
        action="store_true",
        help=(
            "Select one of three frozen row/protection shapes from an "
            "output-blind query/document lexical sketch. The controller "
            "never consumes dataset identity, gold labels, or model output."
        ),
    )
    parser.add_argument("--generalized-strong-active-ratio", type=float, default=0.15)
    parser.add_argument("--generalized-medium-active-ratio", type=float, default=0.20)
    parser.add_argument("--generalized-diffuse-active-ratio", type=float, default=0.25)
    parser.add_argument(
        "--query-protection-tokens",
        type=int,
        default=8192,
        help=(
            "Output-blind Top1-document token/local-head protection budget; "
            "must be a 512-token multiple within one document."
        ),
    )
    parser.add_argument(
        "--row-sparse-segment-cap-ratio", type=float, default=0.50
    )
    parser.add_argument("--row-sparse-checkpoint-stride", type=int, default=512)
    parser.add_argument(
        "--row-sparse-checkpoint-max-islands", type=int, default=64
    )
    parser.add_argument(
        "--radix-eviction-policy",
        choices=("lru", "lfu", "slru", "priority"),
        default="lfu",
        help=(
            "Device radix-tree eviction policy. Prefix materialization "
            "requires LFU so the fixed offline prefix survives interleaved "
            "one-shot dense traffic."
        ),
    )
    parser.add_argument("--ttft-warmup", type=int, default=0)
    parser.add_argument("--ttft-iters", type=int, default=1)
    parser.add_argument(
        "--ttft-target-speedup",
        type=float,
        default=4.0,
        help="Explicit TTFT qualification target; default is the 4x core goal.",
    )
    parser.add_argument(
        "--native-indexer-doc-cap",
        type=int,
        default=0,
        help=(
            "Opt-in per-offline-document C4 Indexer cap. Zero disables the "
            "candidate; online suffix positions are always uncapped."
        ),
    )
    parser.add_argument("--native-indexer-documents", type=int, default=8)
    parser.add_argument(
        "--geometry-template-cache",
        action="store_true",
        help=(
            "Preload immutable fixed-corpus restore geometry after the first "
            "exact request; dynamic slots, leases and TP commits are rebuilt."
        ),
    )
    parser.add_argument(
        "--restore-pipeline-group-layers",
        type=int,
        default=0,
        help=(
            "Qualification-only layer-group restore pipeline; 0 keeps the "
            "certified monolithic three-family launch."
        ),
    )
    parser.add_argument(
        "--merged-prefill-tokens",
        type=int,
        default=0,
        help=(
            "Restore-only physical prefill group; 0 keeps 8K microforwards, "
            "32768 merges four certified segments, and 65536 merges eight."
        ),
    )
    parser.add_argument("--measure-qps", action="store_true")
    parser.add_argument(
        "--qps-concurrencies",
        default="1,2,4,8,16",
        help="Comma-separated closed-loop QPS concurrency points.",
    )
    parser.add_argument("--qps-warmup-waves", type=int, default=0)
    parser.add_argument("--qps-waves", type=int, default=1)
    parser.add_argument("--strict-performance", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--min-token-agreement", type=float, default=0.90)
    parser.add_argument("--min-gpu-free-mib", type=int, default=70000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.60)
    args = parser.parse_args()

    if args.qualification_profile and args.mode != "redknot":
        raise ValueError("--qualification-profile is valid only in RedKnot mode")
    if args.qualification_profile and args.data_manifest:
        raise ValueError(
            "--qualification-profile already freezes its data manifest; "
            "--data-manifest must be omitted"
        )
    if args.qualification_profile and args.prompt_manifest_out:
        raise ValueError(
            "--qualification-profile already freezes its prompt manifest; "
            "--prompt-manifest-out must be omitted"
        )

    if not 0.0 < args.token_sparse_mass <= 1.0:
        raise ValueError("--token-sparse-mass must be in (0, 1]")
    if not math.isfinite(args.ttft_target_speedup) or args.ttft_target_speedup <= 1.0:
        raise ValueError("--ttft-target-speedup must be finite and > 1")
    if args.native_indexer_doc_cap < 0:
        raise ValueError("--native-indexer-doc-cap cannot be negative")
    if args.native_indexer_doc_cap and args.mode != "redknot":
        raise ValueError("native Indexer document bucketing is RedKnot-only")
    if args.native_indexer_documents <= 0:
        raise ValueError("--native-indexer-documents must be positive")
    if not 0.0 < args.token_sparse_deep_mass <= 1.0:
        raise ValueError("--token-sparse-deep-mass must be in (0, 1]")
    if not 0.0 <= args.token_sparse_min_full_ratio <= 1.0:
        raise ValueError("--token-sparse-min-full-ratio must be in [0, 1]")
    if not 0.0 <= args.token_sparse_max_full_ratio <= 1.0:
        raise ValueError("--token-sparse-max-full-ratio must be in [0, 1]")
    if args.token_sparse_min_full_ratio > args.token_sparse_max_full_ratio:
        raise ValueError("token-sparse min full ratio exceeds max full ratio")
    if (
        args.token_sparse_block_tokens < 0
        or (
            args.token_sparse_block_tokens
            and args.token_sparse_block_tokens % 128 != 0
        )
    ):
        raise ValueError(
            "--token-sparse-block-tokens must be 0 or a positive multiple of 128"
        )
    if args.token_sparse_freeze_blocks and not args.token_sparse_block_tokens:
        raise ValueError(
            "--token-sparse-freeze-blocks requires --token-sparse-block-tokens"
        )
    if args.token_sparse_ffn and args.mode != "redknot":
        raise ValueError("token-sparse FFN is valid only for RedKnot qualification")
    if not 0 <= args.restore_pipeline_group_layers <= 37:
        raise ValueError("--restore-pipeline-group-layers must be in [0, 37]")
    if args.restore_pipeline_group_layers and args.mode != "redknot":
        raise ValueError("restore pipeline is valid only for RedKnot qualification")
    if args.row_sparse_online and args.mode != "redknot":
        raise ValueError("row-sparse qualification is valid only for RedKnot")
    if args.combined_headsplit_row_sparse and args.mode != "redknot":
        raise ValueError("combined headsplit/row-sparse is valid only for RedKnot")
    if args.row_sparse_online and args.combined_headsplit_row_sparse:
        raise ValueError(
            "standalone row-sparse and combined headsplit/row-sparse are "
            "mutually exclusive"
        )
    if args.generalized_adaptive_controller:
        if not args.combined_headsplit_row_sparse:
            raise ValueError(
                "--generalized-adaptive-controller requires "
                "--combined-headsplit-row-sparse"
            )
        if int(REDKNOT_TARGET_PROFILES[args.target_tokens]["chunk_tokens"]) < 32768:
            raise ValueError(
                "generalized adaptive controller v1 requires 32K documents"
            )
    generalized_ratios = (
        args.generalized_strong_active_ratio,
        args.generalized_medium_active_ratio,
        args.generalized_diffuse_active_ratio,
    )
    if any(
        not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 0.85
        for ratio in generalized_ratios
    ):
        raise ValueError("generalized active ratios must be finite and in (0, 0.85)")
    if tuple(sorted(generalized_ratios)) != generalized_ratios:
        raise ValueError(
            "generalized active ratios must satisfy strong <= medium <= diffuse"
        )
    if not 0.0 < args.row_sparse_active_ratio < 0.85:
        raise ValueError("--row-sparse-active-ratio must be in (0, 0.85)")
    if (
        args.query_protection_tokens < 512
        or args.query_protection_tokens % 512 != 0
        or args.query_protection_tokens
        > int(REDKNOT_TARGET_PROFILES[args.target_tokens]["chunk_tokens"])
    ):
        raise ValueError(
            "--query-protection-tokens must be a 512-token multiple within "
            "one document"
        )
    if not 0.0 < args.row_sparse_segment_cap_ratio <= 1.0:
        raise ValueError(
            "--row-sparse-segment-cap-ratio must be in (0, 1]"
        )
    if (
        args.row_sparse_checkpoint_stride < 512
        or args.row_sparse_checkpoint_stride % 512 != 0
    ):
        raise ValueError(
            "--row-sparse-checkpoint-stride must be a multiple of 512"
        )
    if not 1 <= args.row_sparse_checkpoint_max_islands <= 64:
        raise ValueError(
            "--row-sparse-checkpoint-max-islands must be in [1, 64]"
        )

    if args.prefix_materialization and args.first_document_prefix:
        raise ValueError(
            "--prefix-materialization and --first-document-prefix are mutually exclusive"
        )

    result, server_log, prompt_manifest, rank_logs = _prepare_output_paths(args)
    if args.mode == "baseline":
        _configure_native(args, result)
    else:
        _configure_redknot(args, result, server_log, prompt_manifest, rank_logs)
    runpy.run_path(str(DRIVER), run_name="__main__")


if __name__ == "__main__":
    main()
