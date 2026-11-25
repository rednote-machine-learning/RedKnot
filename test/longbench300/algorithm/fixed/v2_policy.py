"""Length-adaptive RedKnot V2.1 / V2.2 / V2.3 policy routing and preset definitions.

Pure Python, CPU-only module with zero accelerator imports so it can be loaded
safely during preflight before GPU ownership release.

Generalization law ("Short-Wide, Long-Tight"):
  - <48K (16K / 32K, recall-limited):
      single-stage 20% body budget, NoSpan (0, 0), Online_Head=64.
  - >=48K (64K / 128K / 256K, distractor-limited):
      asymmetric intra-document C2 Span [-8, +24] (configurable via span_min_tokens),
      Online_Head=64 (<96K) / 128 (>=96K), and optional L2->L14 two-stage progressive
      selection with automatic 256K (>=192K) pinned host residual offload.
  - Legacy `--mode s1.6`:
      100% bit-exact original Cone15 15% path (no MinimalCone, no suffix-only,
      no Online_Head, no Span, no Opt-1/2/3, no two-stage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

VALID_BANDS: Tuple[str, ...] = ("16k", "32k", "64k", "128k", "256k")
VALID_MODES: Tuple[str, ...] = ("v2.1-fast", "s1.6")
TWO_STAGE_PRESETS: Dict[str, Tuple[float, float]] = {
    # preset_name: (stage1_fraction, target_fraction) for >=48K
    "short20_long15": (0.12, 0.15),  # Eq-12p3 (H200 500Q accuracy-speed champion)
    "v2.3_12": (0.09, 0.12),         # Eq-9p3  (extreme speed while beating Full)
    "v2.3_10": (0.07, 0.10),         # Eq-7p3  (faster than S1.6 across 16K-256K)
    "m16": (0.16, 0.20),             # Eq-16p4 (20% equal-budget high generalization)
}
VALID_PRESETS: Tuple[str, ...] = ("v2.1-fast", *TWO_STAGE_PRESETS.keys(), "s1.6")

SHORT_LONG_THRESHOLD = 48_000        # <48K stays single-stage 20% + NoSpan.
DEFAULT_SPAN_MIN_TOKENS = 48_000     # Default H200 V2.1/V2.2/V2.3 span cutoff (96_000 for B300 Policy B).
HEAD128_THRESHOLD = 96_000           # Midpoint between 64K and 128K frames.
HOST_OFFLOAD_THRESHOLD = 192_000     # Automatic 256K pinned host offload cutoff.
DEFAULT_SPAN_WINDOW: Tuple[int, int] = (8, 24)  # C2 keys: [-8, +24] -> [-16, +48] tokens.
SHORT_DEFAULT_FRACTION = 0.20
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
        return self.mode == "s1.6"

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

    mode: str = "v2.1-fast"
    two_stage_l14: bool = False
    preset: Optional[str] = None
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS
    host_prefix_offload: Optional[bool] = None
    audit_halo: bool = False

    def __post_init__(self) -> None:
        require(self.mode in VALID_MODES,
                f"Unsupported RedKnot mode {self.mode!r}; expected one of {VALID_MODES}")
        require(type(self.two_stage_l14) is bool, "two_stage_l14 must be a bool")
        require(type(self.span_min_tokens) is int and self.span_min_tokens > 0,
                f"span_min_tokens must be a positive int, got {self.span_min_tokens!r}")
        require(self.host_prefix_offload is None or type(self.host_prefix_offload) is bool,
                "host_prefix_offload must be None or bool")
        require(type(self.audit_halo) is bool, "audit_halo must be a bool")

        preset = self.preset
        if self.mode == "s1.6":
            require(not self.two_stage_l14, "Legacy --mode s1.6 cannot be combined with --two-stage-l14")
            require(preset in (None, "s1.6"),
                    f"Legacy --mode s1.6 cannot be combined with V2 preset {preset!r}")
            object.__setattr__(self, "preset", "s1.6")
            return

        if preset is None:
            resolved_preset = "short20_long15" if self.two_stage_l14 else "v2.1-fast"
            object.__setattr__(self, "preset", resolved_preset)
            return

        require(preset in VALID_PRESETS,
                f"Unsupported RedKnot preset {preset!r}; expected one of {VALID_PRESETS}")
        if preset == "s1.6":
            require(not self.two_stage_l14, "Preset 's1.6' cannot be combined with --two-stage-l14")
            object.__setattr__(self, "mode", "s1.6")
        elif preset == "v2.1-fast":
            require(not self.two_stage_l14,
                    "Single-stage preset 'v2.1-fast' cannot be combined with --two-stage-l14")
        else:
            # Two-stage preset selected explicitly.
            require(preset in TWO_STAGE_PRESETS, f"Unknown two-stage preset {preset!r}")
            object.__setattr__(self, "two_stage_l14", True)

    @classmethod
    def legacy_s16(cls) -> "V2PolicyConfig":
        return cls(mode="s1.6", two_stage_l14=False, preset="s1.6")

    @classmethod
    def default(cls) -> "V2PolicyConfig":
        return cls(mode="v2.1-fast", two_stage_l14=False, preset="v2.1-fast")

    @classmethod
    def from_cli(
        cls,
        *,
        mode: str = "v2.1-fast",
        two_stage_l14: bool = False,
        preset: Optional[str] = None,
        span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
        host_prefix_offload: Union[str, bool, None] = "auto",
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
            mode=mode,
            two_stage_l14=bool(two_stage_l14),
            preset=preset,
            span_min_tokens=int(span_min_tokens),
            host_prefix_offload=offload,
            audit_halo=bool(audit_halo),
        )

    @property
    def is_legacy_s16(self) -> bool:
        return self.mode == "s1.6"

    def resolve(self, input_tokens: int) -> V2Policy:
        """Resolve concrete per-request V2Policy from prompt token count."""
        band = band_from_input_tokens(input_tokens)
        if self.mode == "s1.6":
            return V2Policy(
                mode="s1.6",
                preset="s1.6",
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
            require(self.preset in TWO_STAGE_PRESETS,
                    f"Two-stage mode requires a two-stage preset, got {self.preset!r}")
            stage1_fraction, target_fraction = TWO_STAGE_PRESETS[self.preset]
            two_stage_active = True
            mode_name = "two-stage-l14"
        else:
            stage1_fraction = SHORT_DEFAULT_FRACTION
            target_fraction = SHORT_DEFAULT_FRACTION
            two_stage_active = False
            mode_name = "v2.1-fast"

        if not two_stage_active:
            offload = False
        elif self.host_prefix_offload is None:
            offload = input_tokens >= HOST_OFFLOAD_THRESHOLD
        else:
            offload = self.host_prefix_offload

        return V2Policy(
            mode=mode_name,
            preset=self.preset or "v2.1-fast",
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
            "schema": "redknot-v2-policy-config-v1",
            "mode": self.mode,
            "two_stage_l14": self.two_stage_l14,
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
        return cls(
            mode=str(data.get("mode", "v2.1-fast")),
            two_stage_l14=bool(data.get("two_stage_l14", False)),
            preset=data.get("preset"),
            span_min_tokens=int(data.get("span_min_tokens", DEFAULT_SPAN_MIN_TOKENS)),
            host_prefix_offload=data.get("host_prefix_offload", None),
            audit_halo=bool(data.get("audit_halo", False)),
        )


def resolve_v2_policy(
    band: str,
    *,
    mode: str = "v2.1-fast",
    two_stage_l14: bool = False,
    preset: Optional[str] = None,
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
        mode=mode,
        two_stage_l14=two_stage_l14,
        preset=preset,
        span_min_tokens=span_min_tokens,
        host_prefix_offload=host_prefix_offload,
        audit_halo=audit_halo,
    )
    return cfg.resolve(representative[band])


def resolve_v2_policy_for_tokens(
    input_tokens: int,
    *,
    mode: str = "v2.1-fast",
    two_stage_l14: bool = False,
    preset: Optional[str] = None,
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
    host_prefix_offload: Optional[bool] = None,
    audit_halo: bool = False,
) -> V2Policy:
    """Resolve V2Policy directly from raw prompt token length."""
    cfg = V2PolicyConfig(
        mode=mode,
        two_stage_l14=two_stage_l14,
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
