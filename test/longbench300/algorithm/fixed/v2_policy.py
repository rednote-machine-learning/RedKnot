"""Length-adaptive RedKnot V2 policy routing using unified `Stage1+Stage2` keep percentages.

CPU-only module (zero torch/triton/deepinfer imports) safe during preflight.

Unified `--keep-pct` syntax (`Stage1+Stage2`, percent of document tokens kept online):
  - `9+3` (default): two-stage optimal — `<48K` single-stage 20% (NoSpan, Head64);
    `>=48K` L2 9% + L14 3% = 12% (Span[-8,+24], Head64/128, auto host offload >=192K).
  - `20+0` (or `20`): single-stage optimal — 20% across all lengths, 0 GiB extra
    PrefixBank VRAM, no host offload.
  - Custom ratios (e.g. `12+3`, `16+4`, `15+0`): work directly from CLI with no code edits.

Internal legacy compatibility:
  - `V2PolicyConfig.legacy_s16()` preserves the original 15% Cone15 path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

DEFAULT_KEEP_PCT = "9+3"
SINGLE_STAGE_KEEP_PCT = "20+0"
LEGACY_S16_SPEC = "s1.6"

SHORT_LONG_THRESHOLD = 48_000
DEFAULT_SPAN_MIN_TOKENS = 48_000
HEAD128_THRESHOLD = 96_000
HOST_OFFLOAD_THRESHOLD = 192_000
DEFAULT_SPAN_WINDOW: Tuple[int, int] = (8, 24)

SHORT_TWO_STAGE_FALLBACK_FRACTION = 0.20
LEGACY_S16_FRACTION = 0.15
TWO_STAGE_MID_LAYER = 14


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _fmt_pct(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def parse_keep_pct(spec: str) -> Tuple[float, float, str]:
    """Parse `Stage1+Stage2` (or single `Stage1`) keep percentage string."""
    _require(isinstance(spec, str) and spec.strip() != "",
             f"Invalid --keep-pct {spec!r}; expected 'Stage1+Stage2' (e.g. '9+3', '20+0', '12+3')")
    parts = [p.strip() for p in spec.strip().split("+")]
    _require(len(parts) in (1, 2) and all(parts),
             f"Invalid --keep-pct {spec!r}; expected 'Stage1+Stage2' (e.g. '9+3', '20+0', '12+3')")
    try:
        s1 = float(parts[0])
        s2 = float(parts[1]) if len(parts) == 2 else 0.0
    except ValueError as exc:
        raise ValueError(f"Invalid --keep-pct {spec!r}; percentages must be numbers") from exc
    _require(math.isfinite(s1) and math.isfinite(s2), f"Non-finite --keep-pct: {spec!r}")
    _require(s1 > 0.0 and s2 >= 0.0 and (s1 + s2) <= 100.0,
             f"Invalid --keep-pct {spec!r}; require Stage1 > 0, Stage2 >= 0, Stage1 + Stage2 <= 100")
    return s1, s2, f"{_fmt_pct(s1)}+{_fmt_pct(s2)}"


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
    keep_pct: str
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
        return self.keep_pct == LEGACY_S16_SPEC

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

    keep_pct: str = DEFAULT_KEEP_PCT
    span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS
    host_prefix_offload: Optional[bool] = None
    audit_halo: bool = False

    def __post_init__(self) -> None:
        if self.keep_pct != LEGACY_S16_SPEC:
            _, _, canonical = parse_keep_pct(self.keep_pct)
            object.__setattr__(self, "keep_pct", canonical)
        _require(type(self.span_min_tokens) is int and self.span_min_tokens > 0,
                 f"span_min_tokens must be a positive int, got {self.span_min_tokens!r}")
        _require(self.host_prefix_offload is None or type(self.host_prefix_offload) is bool,
                 "host_prefix_offload must be None or bool")
        _require(type(self.audit_halo) is bool, "audit_halo must be a bool")

    @property
    def is_legacy_s16(self) -> bool:
        return self.keep_pct == LEGACY_S16_SPEC

    @property
    def stage1_pct(self) -> float:
        return 15.0 if self.is_legacy_s16 else parse_keep_pct(self.keep_pct)[0]

    @property
    def stage2_pct(self) -> float:
        return 0.0 if self.is_legacy_s16 else parse_keep_pct(self.keep_pct)[1]

    @property
    def two_stage_l14(self) -> bool:
        return not self.is_legacy_s16 and self.stage2_pct > 0.0

    @classmethod
    def default(cls) -> "V2PolicyConfig":
        return cls(keep_pct=DEFAULT_KEEP_PCT)

    @classmethod
    def legacy_s16(cls) -> "V2PolicyConfig":
        return cls(keep_pct=LEGACY_S16_SPEC)

    @classmethod
    def from_cli(
        cls,
        *,
        keep_pct: str = DEFAULT_KEEP_PCT,
        host_prefix_offload: Union[str, bool, None] = "auto",
        span_min_tokens: int = DEFAULT_SPAN_MIN_TOKENS,
        audit_halo: bool = False,
    ) -> "V2PolicyConfig":
        offload_map = {None: None, "auto": None, True: True, "on": True, False: False, "off": False}
        _require(host_prefix_offload in offload_map, f"Invalid host_prefix_offload: {host_prefix_offload!r}")
        return cls(
            keep_pct=keep_pct,
            span_min_tokens=int(span_min_tokens),
            host_prefix_offload=offload_map[host_prefix_offload],
            audit_halo=bool(audit_halo),
        )

    def resolve(self, input_tokens: int) -> V2Policy:
        band = band_from_input_tokens(input_tokens)
        if self.is_legacy_s16:
            return V2Policy(
                mode=LEGACY_S16_SPEC, keep_pct=LEGACY_S16_SPEC, band=band,
                input_tokens=input_tokens, fraction=LEGACY_S16_FRACTION,
                stage1_fraction=LEGACY_S16_FRACTION, head_width=0, window=(0, 0),
                span_min_tokens=self.span_min_tokens, minimal_cone=False,
                suffix_only=False, gpu_select=False, static_metadata_cache=False,
                direct_halo_copy=False, two_stage_l14=False,
            )
        s1_pct, s2_pct, _ = parse_keep_pct(self.keep_pct)
        if s2_pct == 0.0:
            active_two_stage = False
            s1_frac = frac = round(s1_pct / 100.0, 6)
        elif input_tokens < SHORT_LONG_THRESHOLD:
            active_two_stage = False
            s1_frac = frac = SHORT_TWO_STAGE_FALLBACK_FRACTION
        else:
            active_two_stage = True
            s1_frac = round(s1_pct / 100.0, 6)
            frac = round((s1_pct + s2_pct) / 100.0, 6)
        offload = (
            False if not active_two_stage
            else (input_tokens >= HOST_OFFLOAD_THRESHOLD if self.host_prefix_offload is None else self.host_prefix_offload)
        )
        return V2Policy(
            mode="two-stage" if active_two_stage else "single-stage",
            keep_pct=self.keep_pct,
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
        return dict(schema="redknot-v2-policy-config-v3", **asdict(self))

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["V2PolicyConfig"]:
        if data is None:
            return None
        _require(isinstance(data, Mapping), "V2 policy config must be a mapping")
        keep_pct = data.get("keep_pct")
        if keep_pct is None:
            keep_pct = LEGACY_S16_SPEC if data.get("mode") == LEGACY_S16_SPEC else DEFAULT_KEEP_PCT
        return cls(
            keep_pct=str(keep_pct),
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
