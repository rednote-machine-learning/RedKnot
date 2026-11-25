"""Length-adaptive RedKnot V2 policy routing for two optimal recipes (CPU-only).

Public open-source recipes (both follow "Short-Wide, Long-Tight"):
  - `two-stage` (default): <48K single-stage 20% (NoSpan, Head64);
    >=48K L2->L14 two-stage 9%+3%=12% (Span[-8,+24], Head64/128, auto pinned
    host residual offload at >=192K).
  - `single-stage`: 20% across all lengths (0 GiB extra PrefixBank VRAM;
    <48K NoSpan/Head64, >=48K Span[-8,+24]/Head64-128).

Internal legacy compatibility:
  - `V2PolicyConfig.legacy_s16()` preserves the original 15% Cone15 path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

PRESET_TWO_STAGE = "two-stage"
PRESET_SINGLE_STAGE = "single-stage"
PRESET_LEGACY_S16 = "s1.6"
PUBLIC_PRESETS: Tuple[str, ...] = (PRESET_TWO_STAGE, PRESET_SINGLE_STAGE)
VALID_PRESETS: Tuple[str, ...] = (PRESET_TWO_STAGE, PRESET_SINGLE_STAGE, PRESET_LEGACY_S16)

SHORT_LONG_THRESHOLD = 48_000
DEFAULT_SPAN_MIN_TOKENS = 48_000
HEAD128_THRESHOLD = 96_000
HOST_OFFLOAD_THRESHOLD = 192_000
DEFAULT_SPAN_WINDOW: Tuple[int, int] = (8, 24)

SINGLE_STAGE_FRACTION = 0.20
TWO_STAGE_STAGE1_FRACTION = 0.09
TWO_STAGE_TARGET_FRACTION = 0.12
LEGACY_S16_FRACTION = 0.15
TWO_STAGE_MID_LAYER = 14


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def band_from_input_tokens(input_tokens: int) -> str:
    _require(type(input_tokens) is int and input_tokens > 0,
             f"input_tokens must be a positive integer, got {input_tokens!r}")
    if input_tokens < 24_000:
        return "16k"
    if input_tokens < SHORT_LONG_THRESHOLD:
        return "32k"
    if input_tokens < HEAD128_THRESHOLD:
        return "64k"
    if input_tokens < HOST_OFFLOAD_THRESHOLD:
        return "128k"
    return "256k"


@dataclass(frozen=True)
class V2Policy:
    """Resolved per-request RedKnot execution policy for a concrete prompt length."""

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

    @property
    def is_legacy_s16(self) -> bool:
        return self.preset == PRESET_LEGACY_S16

    @property
    def is_v2(self) -> bool:
        return not self.is_legacy_s16

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["window"] = list(self.window)
        return out


@dataclass(frozen=True)
class V2PolicyConfig:
    """Request-independent policy config carried through CLI and runtime cfg."""

    preset: str = PRESET_TWO_STAGE
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS
    host_prefix_offload: Optional[bool] = None
    audit_halo: bool = False

    def __post_init__(self) -> None:
        _require(self.preset in VALID_PRESETS,
                 f"Unsupported RedKnot preset {self.preset!r}; expected one of {PUBLIC_PRESETS}")
        _require(type(self.span_min_tokens) is int and self.span_min_tokens > 0,
                 f"span_min_tokens must be a positive int, got {self.span_min_tokens!r}")
        _require(self.host_prefix_offload is None or type(self.host_prefix_offload) is bool,
                 "host_prefix_offload must be None or bool")
        _require(type(self.audit_halo) is bool, "audit_halo must be a bool")

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
        offload_map = {None: None, "auto": None, True: True, "on": True, False: False, "off": False}
        _require(host_prefix_offload in offload_map, f"Invalid host_prefix_offload: {host_prefix_offload!r}")
        return cls(
            preset=preset,
            span_min_tokens=int(span_min_tokens),
            host_prefix_offload=offload_map[host_prefix_offload],
            audit_halo=bool(audit_halo),
        )

    def resolve(self, input_tokens: int) -> V2Policy:
        band = band_from_input_tokens(input_tokens)
        if self.is_legacy_s16:
            return V2Policy(
                mode=PRESET_LEGACY_S16, preset=PRESET_LEGACY_S16, band=band,
                input_tokens=input_tokens, fraction=LEGACY_S16_FRACTION,
                stage1_fraction=LEGACY_S16_FRACTION, head_width=0, window=(0, 0),
                span_min_tokens=self.span_min_tokens, minimal_cone=False,
                suffix_only=False, gpu_select=False, static_metadata_cache=False,
                direct_halo_copy=False, two_stage_l14=False,
            )
        active_two_stage = self.two_stage_l14 and input_tokens >= SHORT_LONG_THRESHOLD
        s1_frac, frac = (
            (TWO_STAGE_STAGE1_FRACTION, TWO_STAGE_TARGET_FRACTION)
            if active_two_stage else (SINGLE_STAGE_FRACTION, SINGLE_STAGE_FRACTION)
        )
        offload = (
            False if not active_two_stage
            else (input_tokens >= HOST_OFFLOAD_THRESHOLD if self.host_prefix_offload is None else self.host_prefix_offload)
        )
        return V2Policy(
            mode=PRESET_TWO_STAGE if active_two_stage else PRESET_SINGLE_STAGE,
            preset=self.preset,
            band=band,
            input_tokens=input_tokens,
            fraction=frac,
            stage1_fraction=s1_frac,
            head_width=128 if input_tokens >= HEAD128_THRESHOLD else 64,
            window=DEFAULT_SPAN_WINDOW if input_tokens >= self.span_min_tokens else (0, 0),
            span_min_tokens=self.span_min_tokens,
            minimal_cone=True,
            suffix_only=True,
            gpu_select=True,
            static_metadata_cache=True,
            direct_halo_copy=True,
            two_stage_l14=active_two_stage,
            host_prefix_offload=offload,
            audit_halo=self.audit_halo,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(schema="redknot-v2-policy-config-v2", **asdict(self))

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["V2PolicyConfig"]:
        if data is None:
            return None
        _require(isinstance(data, Mapping), "V2 policy config must be a mapping")
        preset = data.get("preset") or (
            PRESET_LEGACY_S16 if data.get("mode") == PRESET_LEGACY_S16 else PRESET_TWO_STAGE
        )
        return cls(
            preset=str(preset),
            span_min_tokens=int(data.get("span_min_tokens", DEFAULT_SPAN_MIN_TOKENS)),
            host_prefix_offload=data.get("host_prefix_offload", None),
            audit_halo=bool(data.get("audit_halo", False)),
        )


def resolve_request_policy(
    policy_or_config: Union[None, V2Policy, V2PolicyConfig, Mapping[str, Any]],
    input_tokens: int,
) -> Optional[V2Policy]:
    if policy_or_config is None or isinstance(policy_or_config, V2Policy):
        return policy_or_config
    if isinstance(policy_or_config, V2PolicyConfig):
        return policy_or_config.resolve(input_tokens)
    if isinstance(policy_or_config, Mapping):
        return V2PolicyConfig.from_dict(policy_or_config).resolve(input_tokens)
    raise ValueError(f"Unsupported V2 policy object: {type(policy_or_config).__name__}")
