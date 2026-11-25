"""Length-adaptive RedKnot V2 policy routing with two optimal recipes.

Pure Python, CPU-only module with zero accelerator imports so it can be loaded
safely during preflight before GPU ownership release.

Two public open-source recipes (both follow "Short-Wide, Long-Tight"):
  1. `two-stage` (default — best overall accuracy, generalization, and speed):
       - <48K (16K / 32K): single-stage 20%, NoSpan (0, 0), Online_Head=64,
         0 GiB extra PrefixBank VRAM.
       - >=48K (64K / 128K / 256K): L2->L14 two-stage 9%+3%=12% (Stage-1 9%,
         L14 promotes +3% multi-hop bridges), Span[-8, +24], Online_Head=64/128,
         auto pinned host offload of L3..L8 residuals at >=192K (256K).
       - 8xH200 benchmark: 500Q 51.85 (vs Full 51.52, +0.33), 900Q 53.63,
         mean TTFT 14.46s (2.43x), 256K 37.61 (vs Full 37.65).
  2. `single-stage` (zero extra PrefixBank VRAM, simplest execution path):
       - All lengths: single-stage 20% body budget, 0 GiB extra cache VRAM.
       - <48K: NoSpan (0, 0), Online_Head=64.
       - >=48K: Span[-8, +24], Online_Head=64 (<96K) / 128 (>=96K).
       - 8xH200 benchmark: 500Q 51.78 (vs Full 51.52, +0.26), 900Q 53.30.

Internal legacy compatibility:
  - `V2PolicyConfig.legacy_s16()` preserves the original 15% Cone15 path
    (for resuming legacy run directories without altering existing code paths).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

VALID_BANDS: Tuple[str, ...] = ("16k", "32k", "64k", "128k", "256k")
PRESET_TWO_STAGE = "two-stage"
PRESET_SINGLE_STAGE = "single-stage"
PRESET_LEGACY_S16 = "s1.6"
PUBLIC_PRESETS: Tuple[str, ...] = (PRESET_TWO_STAGE, PRESET_SINGLE_STAGE)
VALID_PRESETS: Tuple[str, ...] = (PRESET_TWO_STAGE, PRESET_SINGLE_STAGE, PRESET_LEGACY_S16)

SHORT_LONG_THRESHOLD = 48_000        # <48K stays single-stage 20% + NoSpan.
DEFAULT_SPAN_MIN_TOKENS = 48_000     # Default H200 Span[-8, +24] cutoff.
HEAD128_THRESHOLD = 96_000           # Midpoint between 64K and 128K frames.
HOST_OFFLOAD_THRESHOLD = 192_000     # Automatic 256K pinned host offload cutoff.
DEFAULT_SPAN_WINDOW: Tuple[int, int] = (8, 24)  # C2 keys: [-8, +24] -> [-16, +48] tokens.

SINGLE_STAGE_FRACTION = 0.20
TWO_STAGE_STAGE1_FRACTION = 0.09
TWO_STAGE_TARGET_FRACTION = 0.12
LEGACY_S16_FRACTION = 0.15
TWO_STAGE_MID_LAYER = 14


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def band_from_input_tokens(input_tokens: int) -> str:
    """Infer frozen LongBench length band from raw prompt token count."""
    require(type(input_tokens) is int and input_tokens > 0,
            f"input_tokens must be a positive integer, got {input_tokens!r}")
    if input_tokens < 24_000:
        return "16k"
    if input_tokens < 48_000:
        return "32k"
    if input_tokens < HEAD128_THRESHOLD:
        return "64k"
    if input_tokens < HOST_OFFLOAD_THRESHOLD:
        return "128k"
    return "256k"


@dataclass(frozen=True)
class V2Policy:
    """Concrete per-request RedKnot execution policy resolved for a prompt length."""

    mode: str
    preset: str
    band: str
    input_tokens: int
    fraction: float
    stage1_fraction: float
    head_width: int
    window: Tuple[int, int]
    span_min_tokens: int
    minimal_cone: bool
    suffix_only: bool
    gpu_select: bool
    static_metadata_cache: bool
    direct_halo_copy: bool
    two_stage_l14: bool
    two_stage_mid_layer: int = TWO_STAGE_MID_LAYER
    host_prefix_offload: bool = False
    audit_halo: bool = False
    bank: str = "plain"
    alpha: float = 1.0
    opts: Tuple[int, ...] = (1, 2, 3)

    @property
    def is_legacy_s16(self) -> bool:
        return self.preset == PRESET_LEGACY_S16

    @property
    def is_v2(self) -> bool:
        return not self.is_legacy_s16

    @property
    def span_enabled(self) -> bool:
        return any(self.window)

    @property
    def online_head_enabled(self) -> bool:
        return self.head_width > 0

    @property
    def direct_copy(self) -> bool:
        return self.direct_halo_copy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "preset": self.preset,
            "band": self.band,
            "input_tokens": self.input_tokens,
            "fraction": self.fraction,
            "stage1_fraction": self.stage1_fraction,
            "head_width": self.head_width,
            "window": list(self.window),
            "span_min_tokens": self.span_min_tokens,
            "minimal_cone": self.minimal_cone,
            "suffix_only": self.suffix_only,
            "gpu_select": self.gpu_select,
            "static_metadata_cache": self.static_metadata_cache,
            "direct_halo_copy": self.direct_halo_copy,
            "two_stage_l14": self.two_stage_l14,
            "two_stage_mid_layer": self.two_stage_mid_layer,
            "host_prefix_offload": self.host_prefix_offload,
            "audit_halo": self.audit_halo,
            "bank": self.bank,
            "alpha": self.alpha,
            "opts": list(self.opts),
        }


@dataclass(frozen=True)
class V2PolicyConfig:
    """Request-independent configuration carried in runtime cfg and CLI options."""

    preset: str = PRESET_TWO_STAGE
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS
    host_prefix_offload: Optional[bool] = None
    audit_halo: bool = False

    def __post_init__(self) -> None:
        require(self.preset in VALID_PRESETS,
                f"Unsupported RedKnot preset {self.preset!r}; expected one of {PUBLIC_PRESETS}")
        require(type(self.span_min_tokens) is int and self.span_min_tokens > 0,
                f"span_min_tokens must be a positive int, got {self.span_min_tokens!r}")
        require(self.host_prefix_offload is None or type(self.host_prefix_offload) is bool,
                "host_prefix_offload must be None or bool")
        require(type(self.audit_halo) is bool, "audit_halo must be a bool")

    @property
    def mode(self) -> str:
        return self.preset

    @property
    def two_stage_l14(self) -> bool:
        return self.preset == PRESET_TWO_STAGE

    @property
    def is_legacy_s16(self) -> bool:
        return self.preset == PRESET_LEGACY_S16

    @classmethod
    def default(cls) -> "V2PolicyConfig":
        return cls(preset=PRESET_TWO_STAGE)

    @classmethod
    def two_stage(
        cls,
        *,
        host_prefix_offload: Optional[bool] = None,
        span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
        audit_halo: bool = False,
    ) -> "V2PolicyConfig":
        return cls(
            preset=PRESET_TWO_STAGE,
            span_min_tokens=span_min_tokens,
            host_prefix_offload=host_prefix_offload,
            audit_halo=audit_halo,
        )

    @classmethod
    def single_stage(
        cls,
        *,
        span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
        audit_halo: bool = False,
    ) -> "V2PolicyConfig":
        return cls(
            preset=PRESET_SINGLE_STAGE,
            span_min_tokens=span_min_tokens,
            host_prefix_offload=False,
            audit_halo=audit_halo,
        )

    @classmethod
    def legacy_s16(cls) -> "V2PolicyConfig":
        return cls(preset=PRESET_LEGACY_S16)

    @classmethod
    def from_cli(
        cls,
        *,
        preset: str = PRESET_TWO_STAGE,
        host_prefix_offload: Union[str, bool, None] = "auto",
        span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
        audit_halo: bool = False,
    ) -> "V2PolicyConfig":
        if host_prefix_offload in (None, "auto"):
            offload: Optional[bool] = None
        elif host_prefix_offload in (True, "on", "true", "1"):
            offload = True
        elif host_prefix_offload in (False, "off", "false", "0"):
            offload = False
        else:
            raise ValueError(f"Invalid host_prefix_offload setting: {host_prefix_offload!r}")
        return cls(
            preset=preset,
            span_min_tokens=int(span_min_tokens),
            host_prefix_offload=offload,
            audit_halo=bool(audit_halo),
        )

    def resolve(self, input_tokens: int) -> V2Policy:
        """Resolve concrete per-request V2Policy from prompt token count."""
        band = band_from_input_tokens(input_tokens)
        if self.is_legacy_s16:
            return V2Policy(
                mode=PRESET_LEGACY_S16,
                preset=PRESET_LEGACY_S16,
                band=band,
                input_tokens=input_tokens,
                fraction=LEGACY_S16_FRACTION,
                stage1_fraction=LEGACY_S16_FRACTION,
                head_width=0,
                window=(0, 0),
                span_min_tokens=self.span_min_tokens,
                minimal_cone=False,
                suffix_only=False,
                gpu_select=False,
                static_metadata_cache=False,
                direct_halo_copy=False,
                two_stage_l14=False,
                two_stage_mid_layer=TWO_STAGE_MID_LAYER,
                host_prefix_offload=False,
                audit_halo=False,
                bank="plain",
                alpha=1.0,
                opts=(),
            )

        head_width = 128 if input_tokens >= HEAD128_THRESHOLD else 64
        window = DEFAULT_SPAN_WINDOW if input_tokens >= self.span_min_tokens else (0, 0)
        long_regime = input_tokens >= SHORT_LONG_THRESHOLD

        if self.two_stage_l14 and long_regime:
            stage1_fraction = TWO_STAGE_STAGE1_FRACTION
            target_fraction = TWO_STAGE_TARGET_FRACTION
            two_stage_active = True
            mode_name = PRESET_TWO_STAGE
        else:
            stage1_fraction = SINGLE_STAGE_FRACTION
            target_fraction = SINGLE_STAGE_FRACTION
            two_stage_active = False
            mode_name = PRESET_SINGLE_STAGE

        if not two_stage_active:
            offload = False
        elif self.host_prefix_offload is None:
            offload = input_tokens >= HOST_OFFLOAD_THRESHOLD
        else:
            offload = self.host_prefix_offload

        return V2Policy(
            mode=mode_name,
            preset=self.preset,
            band=band,
            input_tokens=input_tokens,
            fraction=target_fraction,
            stage1_fraction=stage1_fraction,
            head_width=head_width,
            window=window,
            span_min_tokens=self.span_min_tokens,
            minimal_cone=True,
            suffix_only=True,
            gpu_select=True,
            static_metadata_cache=True,
            direct_halo_copy=True,
            two_stage_l14=two_stage_active,
            two_stage_mid_layer=TWO_STAGE_MID_LAYER,
            host_prefix_offload=offload,
            audit_halo=self.audit_halo,
            bank="plain",
            alpha=1.0,
            opts=(1, 2, 3),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "redknot-v2-policy-config-v2",
            "preset": self.preset,
            "span_min_tokens": self.span_min_tokens,
            "host_prefix_offload": self.host_prefix_offload,
            "audit_halo": self.audit_halo,
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["V2PolicyConfig"]:
        if data is None:
            return None
        require(isinstance(data, Mapping), "V2 policy config must be a mapping")
        preset = data.get("preset")
        if preset is None:
            preset = PRESET_LEGACY_S16 if data.get("mode") == PRESET_LEGACY_S16 else PRESET_TWO_STAGE
        return cls(
            preset=str(preset),
            span_min_tokens=int(data.get("span_min_tokens", DEFAULT_SPAN_MIN_TOKENS)),
            host_prefix_offload=data.get("host_prefix_offload", None),
            audit_halo=bool(data.get("audit_halo", False)),
        )


def resolve_v2_policy(
    band: str,
    *,
    preset: str = PRESET_TWO_STAGE,
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
    host_prefix_offload: Optional[bool] = None,
    audit_halo: bool = False,
) -> V2Policy:
    """Resolve V2Policy from a canonical length band name."""
    representative = {
        "16k": 16_000,
        "32k": 32_000,
        "64k": 64_000,
        "128k": 128_000,
        "256k": 250_000,
    }
    require(band in representative,
            f"Unsupported length band {band!r}; expected one of {VALID_BANDS}")
    cfg = V2PolicyConfig(
        preset=preset,
        span_min_tokens=span_min_tokens,
        host_prefix_offload=host_prefix_offload,
        audit_halo=audit_halo,
    )
    return cfg.resolve(representative[band])


def resolve_v2_policy_for_tokens(
    input_tokens: int,
    *,
    preset: str = PRESET_TWO_STAGE,
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
    host_prefix_offload: Optional[bool] = None,
    audit_halo: bool = False,
) -> V2Policy:
    """Resolve V2Policy directly from raw prompt token length."""
    cfg = V2PolicyConfig(
        preset=preset,
        span_min_tokens=span_min_tokens,
        host_prefix_offload=host_prefix_offload,
        audit_halo=audit_halo,
    )
    return cfg.resolve(input_tokens)


def resolve_request_policy(
    policy_or_config: Union[None, V2Policy, V2PolicyConfig, Mapping[str, Any]],
    input_tokens: int,
) -> Optional[V2Policy]:
    """Normalize a controller policy configuration into a concrete request V2Policy."""
    if policy_or_config is None:
        return None
    if isinstance(policy_or_config, V2Policy):
        return policy_or_config
    if isinstance(policy_or_config, V2PolicyConfig):
        return policy_or_config.resolve(input_tokens)
    if isinstance(policy_or_config, Mapping):
        return V2PolicyConfig.from_dict(policy_or_config).resolve(input_tokens)
    raise ValueError(f"Unsupported V2 policy object: {type(policy_or_config).__name__}")
