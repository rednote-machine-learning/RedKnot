"""Paired projected-output drift profiling for DeepSeek V4 MLA heads.

This module deliberately has no SGLang or serving dependency.  It operates on
the position-independent per-head attention output immediately before
DeepSeek V4's grouped ``wo_a`` projection.  A caller can therefore use it from
an offline calibration harness, a model hook, or focused CPU tests.

The safety model is conservative:

* every expected ``(layer, segment, oracle)`` pair must be present;
* snapshot and oracle rows must match exactly;
* head selection is balanced across contiguous tensor-parallel head shards;
* subset admission uses the sum of individual head error norms.  Negative
  off-diagonal Gram entries are reported, but never used as error cancellation
  to admit more local heads;
* dense-prefix layers can never become local.

The returned head-config dictionary uses the existing
``redknot_deepseek_v4_mla_head_config_v1`` layout and may be loaded directly by
``DeepSeekV4MLAHeadConfig.from_json``.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


DRIFT_REPORT_FORMAT = "redknot_deepseek_v4_mla_head_drift_v1"
HEAD_CONFIG_FORMAT = "redknot_deepseek_v4_mla_head_config_v1"
STRICT_SELECTION_MODE = "strict_thresholded"
EXPLORATORY_FIXED_COUNT_SELECTION_MODE = "exploratory_fixed_count"


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def decompose_wo_a_per_head(
    per_head_output: torch.Tensor,
    wo_a_weight: torch.Tensor,
) -> torch.Tensor:
    """Return each logical head's exact contribution to grouped ``wo_a``.

    Parameters
    ----------
    per_head_output:
        Tensor shaped ``[tokens, heads, head_dim]`` after inverse RoPE.
    wo_a_weight:
        Grouped linear weight shaped
        ``[output_groups, o_lora_rank, group_input_dim]``.  Heads are contiguous
        inside each output group, matching DeepSeek V4's existing
        ``o.view(tokens, output_groups, -1)`` implementation.

    Returns
    -------
    Tensor shaped ``[tokens, heads, o_lora_rank]``.  Reshaping it to
    ``[tokens, output_groups, heads_per_group, o_lora_rank]`` and summing over
    ``heads_per_group`` is exactly equivalent to the grouped ``wo_a`` einsum.

    Notes
    -----
    The function preserves autograd and the input device.  Accumulation follows
    PyTorch's einsum dtype semantics; callers that need profiling in FP32 should
    pass FP32 tensors explicitly.
    """

    if not torch.is_tensor(per_head_output) or not torch.is_tensor(wo_a_weight):
        raise TypeError("per_head_output and wo_a_weight must be torch tensors")
    if per_head_output.ndim != 3:
        raise ValueError(
            "per_head_output must have shape [tokens, heads, head_dim], got "
            f"{tuple(per_head_output.shape)}"
        )
    if wo_a_weight.ndim != 3:
        raise ValueError(
            "wo_a_weight must have shape [groups, rank, group_input_dim], got "
            f"{tuple(wo_a_weight.shape)}"
        )
    if per_head_output.device != wo_a_weight.device:
        raise ValueError("per_head_output and wo_a_weight must be on the same device")
    if per_head_output.dtype != wo_a_weight.dtype:
        raise ValueError("per_head_output and wo_a_weight must have the same dtype")

    tokens, num_heads, head_dim = per_head_output.shape
    num_groups, o_lora_rank, group_input_dim = wo_a_weight.shape
    if tokens <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ValueError("per_head_output dimensions must all be positive")
    if num_groups <= 0 or o_lora_rank <= 0 or group_input_dim <= 0:
        raise ValueError("wo_a_weight dimensions must all be positive")
    if num_heads % num_groups != 0:
        raise ValueError(
            f"heads={num_heads} must be divisible by output_groups={num_groups}"
        )
    heads_per_group = num_heads // num_groups
    expected_group_input = heads_per_group * head_dim
    if group_input_dim != expected_group_input:
        raise ValueError(
            "wo_a group_input_dim does not match contiguous head layout: "
            f"got {group_input_dim}, expected {heads_per_group} * {head_dim} "
            f"= {expected_group_input}"
        )

    grouped_output = per_head_output.reshape(
        tokens, num_groups, heads_per_group, head_dim
    )
    grouped_weight = wo_a_weight.reshape(
        num_groups, o_lora_rank, heads_per_group, head_dim
    )
    contributions = torch.einsum("tghd,grhd->tghr", grouped_output, grouped_weight)
    return contributions.reshape(tokens, num_heads, o_lora_rank)


def sum_wo_a_head_contributions(
    per_head_projection: torch.Tensor,
    *,
    num_output_groups: int,
) -> torch.Tensor:
    """Sum ``decompose_wo_a_per_head`` output back to grouped ``wo_a``."""

    if not torch.is_tensor(per_head_projection) or per_head_projection.ndim != 3:
        raise ValueError("per_head_projection must have shape [tokens, heads, rank]")
    groups = _strict_int("num_output_groups", num_output_groups, minimum=1)
    tokens, heads, rank = per_head_projection.shape
    if heads <= 0 or heads % groups != 0:
        raise ValueError(
            f"heads={heads} must be positive and divisible by groups={groups}"
        )
    return per_head_projection.reshape(tokens, groups, heads // groups, rank).sum(dim=2)


@dataclass(frozen=True)
class HeadDriftMetrics:
    """Accuracy-drift statistics for one logical head."""

    rms: float
    relative_rms: float
    row_p95: float
    row_p99: float
    row_max: float
    cosine: float
    count_rows: int
    count_values: int
    worst_pair_relative_rms: float
    worst_pair_row_p99: float
    worst_pair_row_max: float
    worst_pair_cosine: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rms": self.rms,
            "relative_rms": self.relative_rms,
            "row_p95": self.row_p95,
            "row_p99": self.row_p99,
            "row_max": self.row_max,
            "cosine": self.cosine,
            "count_rows": self.count_rows,
            "count_values": self.count_values,
            "worst_pair_relative_rms": self.worst_pair_relative_rms,
            "worst_pair_row_p99": self.worst_pair_row_p99,
            "worst_pair_row_max": self.worst_pair_row_max,
            "worst_pair_cosine": self.worst_pair_cosine,
        }


@dataclass(frozen=True)
class DriftPairMetrics:
    """Per-head statistics for one matched snapshot/oracle pair."""

    layer_id: int
    segment_id: str
    oracle_id: str
    row_count: int
    row_ids_sha256: str
    head_metrics: Tuple[HeadDriftMetrics, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "segment_id": self.segment_id,
            "oracle_id": self.oracle_id,
            "row_count": self.row_count,
            "row_ids_sha256": self.row_ids_sha256,
            "heads": [item.to_dict() for item in self.head_metrics],
        }


@dataclass(frozen=True)
class MLAHeadDriftReport:
    """Final immutable report used by the TP-balanced selector."""

    num_layers: int
    num_heads: int
    num_output_groups: int
    tp_size: int
    tp_world_size: int
    represented_tp_ranks: Tuple[int, ...]
    calibration_id: str
    dense_prefix_layers: int
    expected_segments: Tuple[str, ...]
    expected_oracles: Tuple[str, ...]
    aggregate_metrics: Tuple[Tuple[HeadDriftMetrics, ...], ...]
    pair_metrics: Tuple[DriftPairMetrics, ...]
    # Raw rank-local delta Gram matrices, [L, TP, H/TP, H/TP].
    rank_gram: torch.Tensor
    # Squared norm of the rank-local summed oracle projection, [L, TP].
    rank_reference_energy: torch.Tensor
    # Per-pair equivalents aligned one-to-one with pair_metrics.  The selector
    # uses the worst pair instead of allowing a high-energy easy pair to dilute
    # a low-energy, high-risk segment/oracle combination.
    pair_rank_gram: torch.Tensor
    pair_rank_reference_energy: torch.Tensor

    @property
    def heads_per_rank(self) -> int:
        return self.num_heads // self.tp_size

    @property
    def groups_per_rank(self) -> int:
        return self.num_output_groups // self.tp_size

    @property
    def heads_per_output_group(self) -> int:
        return self.num_heads // self.num_output_groups

    def head_no_cancel_cost(self, layer_id: int, head_id: int) -> float:
        """Return an individual error norm normalized to its TP-rank oracle.

        These costs are additive in the selector.  Summing them is a triangle-
        inequality bound and therefore cannot benefit from negative covariance
        between two head errors.
        """

        layer = _strict_int("layer_id", layer_id)
        head = _strict_int("head_id", head_id)
        if layer >= self.num_layers or head >= self.num_heads:
            raise IndexError("layer_id or head_id is outside the report")
        rank = head // self.heads_per_rank
        local_head = head % self.heads_per_rank
        costs = []
        for pair_index, pair in enumerate(self.pair_metrics):
            if pair.layer_id != layer:
                continue
            diagonal = float(
                self.pair_rank_gram[pair_index, rank, local_head, local_head].item()
            )
            reference = float(self.pair_rank_reference_energy[pair_index, rank].item())
            if diagonal <= 0.0:
                costs.append(0.0)
            elif reference <= 0.0:
                costs.append(float("inf"))
            else:
                costs.append(math.sqrt(diagonal / reference))
        if not costs:
            raise RuntimeError(f"layer {layer} has no pair-level drift evidence")
        return max(costs)

    def normalized_rank_gram(self) -> torch.Tensor:
        """Return observed Gram/reference energy for diagnostics only.

        The selector intentionally does not use negative entries from this
        tensor to justify additional local heads.
        """

        denom = self.rank_reference_energy[..., None, None]
        inf = torch.full_like(self.rank_gram, float("inf"))
        return torch.where(denom > 0, self.rank_gram / denom, inf)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": DRIFT_REPORT_FORMAT,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_output_groups": self.num_output_groups,
            "tp_size": self.tp_size,
            "tp_world_size": self.tp_world_size,
            "represented_tp_ranks": list(self.represented_tp_ranks),
            "calibration_id": self.calibration_id,
            "dense_prefix_layers": self.dense_prefix_layers,
            "expected_segments": list(self.expected_segments),
            "expected_oracles": list(self.expected_oracles),
            "aggregate_heads": [
                [item.to_dict() for item in layer] for layer in self.aggregate_metrics
            ],
            "pairs": [item.to_dict() for item in self.pair_metrics],
            "rank_gram": self.rank_gram.detach().cpu().tolist(),
            "rank_reference_energy": (
                self.rank_reference_energy.detach().cpu().tolist()
            ),
            "pair_rank_gram": self.pair_rank_gram.detach().cpu().tolist(),
            "pair_rank_reference_energy": (
                self.pair_rank_reference_energy.detach().cpu().tolist()
            ),
            "selector_safety": ("worst_pair_additive_head_norm_no_error_cancellation"),
        }


@dataclass(frozen=True)
class _ProjectionRows:
    row_ids: torch.Tensor
    values: torch.Tensor


def _safe_ratio_sqrt(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    zeros = numerator <= 0
    safe = torch.sqrt(
        numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    )
    inf = torch.full_like(safe, float("inf"))
    return torch.where(
        zeros, torch.zeros_like(safe), torch.where(denominator > 0, safe, inf)
    )


def _row_relative_error(
    snapshot: torch.Tensor,
    oracle: torch.Tensor,
) -> torch.Tensor:
    delta_norm = torch.linalg.vector_norm(snapshot - oracle, dim=-1)
    oracle_norm = torch.linalg.vector_norm(oracle, dim=-1)
    tiny = torch.finfo(delta_norm.dtype).tiny
    ratio = delta_norm / oracle_norm.clamp_min(tiny)
    return torch.where(
        delta_norm <= 0,
        torch.zeros_like(ratio),
        torch.where(oracle_norm > 0, ratio, torch.full_like(ratio, float("inf"))),
    )


def _metrics_from_tensors(
    snapshot: torch.Tensor,
    oracle: torch.Tensor,
) -> Tuple[HeadDriftMetrics, ...]:
    if snapshot.shape != oracle.shape or snapshot.ndim != 3:
        raise ValueError("snapshot and oracle must share [rows, heads, rank] shape")
    snap = snapshot.float()
    ref = oracle.float()
    delta = snap - ref
    rows, heads, rank = delta.shape
    delta_energy = delta.square().sum(dim=(0, 2)).double()
    oracle_energy = ref.square().sum(dim=(0, 2)).double()
    snapshot_energy = snap.square().sum(dim=(0, 2)).double()
    dot = (snap * ref).sum(dim=(0, 2)).double()
    count_values = rows * rank
    rms = torch.sqrt(delta_energy / max(1, count_values))
    relative = _safe_ratio_sqrt(delta_energy, oracle_energy)
    denom = torch.sqrt(snapshot_energy * oracle_energy)
    cosine = torch.where(
        (snapshot_energy <= 0) & (oracle_energy <= 0),
        torch.ones_like(dot),
        torch.where(denom > 0, dot / denom, torch.zeros_like(dot)),
    ).clamp(-1.0, 1.0)
    row_relative = _row_relative_error(snap, ref)
    p95 = torch.quantile(row_relative, 0.95, dim=0)
    p99 = torch.quantile(row_relative, 0.99, dim=0)
    row_max = row_relative.max(dim=0).values

    out = []
    for head in range(heads):
        rel = float(relative[head].item())
        hp99 = float(p99[head].item())
        hmax = float(row_max[head].item())
        hcos = float(cosine[head].item())
        out.append(
            HeadDriftMetrics(
                rms=float(rms[head].item()),
                relative_rms=rel,
                row_p95=float(p95[head].item()),
                row_p99=hp99,
                row_max=hmax,
                cosine=hcos,
                count_rows=rows,
                count_values=count_values,
                worst_pair_relative_rms=rel,
                worst_pair_row_p99=hp99,
                worst_pair_row_max=hmax,
                worst_pair_cosine=hcos,
            )
        )
    return tuple(out)


def _aggregate_metrics(
    snapshots: Sequence[torch.Tensor],
    oracles: Sequence[torch.Tensor],
    pair_metrics: Sequence[Tuple[HeadDriftMetrics, ...]],
) -> Tuple[HeadDriftMetrics, ...]:
    snapshot = torch.cat(tuple(snapshots), dim=0)
    oracle = torch.cat(tuple(oracles), dim=0)
    aggregate = list(_metrics_from_tensors(snapshot, oracle))
    heads = snapshot.shape[1]
    out = []
    for head in range(heads):
        base = aggregate[head]
        pairs = [items[head] for items in pair_metrics]
        out.append(
            HeadDriftMetrics(
                rms=base.rms,
                relative_rms=base.relative_rms,
                row_p95=base.row_p95,
                row_p99=base.row_p99,
                row_max=base.row_max,
                cosine=base.cosine,
                count_rows=base.count_rows,
                count_values=base.count_values,
                worst_pair_relative_rms=max(x.relative_rms for x in pairs),
                worst_pair_row_p99=max(x.row_p99 for x in pairs),
                worst_pair_row_max=max(x.row_max for x in pairs),
                worst_pair_cosine=min(x.cosine for x in pairs),
            )
        )
    return tuple(out)


class MLAHeadDriftCollector:
    """Collect exact paired snapshot/oracle projected-output observations.

    ``expected_segments`` and ``expected_oracles`` are mandatory and non-empty.
    ``finalize`` rejects missing pairs, extra pairs, row mismatches, non-finite
    projections, and inconsistent projection rank.  Row IDs are local positions
    within a segment and must be CPU integer tensors; projection values may be
    CPU or GPU tensors.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        tp_size: int,
        num_output_groups: Optional[int] = None,
        tp_world_size: Optional[int] = None,
        represented_tp_ranks: Optional[Sequence[int]] = None,
        calibration_id: str = "",
        dense_prefix_layers: int,
        expected_segments: Sequence[str],
        expected_oracles: Sequence[str],
    ) -> None:
        self.num_layers = _strict_int("num_layers", num_layers, minimum=1)
        self.num_heads = _strict_int("num_heads", num_heads, minimum=1)
        self.tp_size = _strict_int("tp_size", tp_size, minimum=1)
        self.num_output_groups = _strict_int(
            "num_output_groups",
            self.tp_size if num_output_groups is None else num_output_groups,
            minimum=1,
        )
        self.tp_world_size = _strict_int(
            "tp_world_size",
            self.tp_size if tp_world_size is None else tp_world_size,
            minimum=1,
        )
        if represented_tp_ranks is None:
            represented_tp_ranks = tuple(range(self.tp_size))
        represented = tuple(
            _strict_int("represented_tp_rank", value) for value in represented_tp_ranks
        )
        if len(represented) != self.tp_size or len(set(represented)) != len(
            represented
        ):
            raise ValueError(
                "represented_tp_ranks must contain one unique rank per "
                "represented TP shard"
            )
        if any(value >= self.tp_world_size for value in represented):
            raise ValueError("represented_tp_ranks contains an out-of-range rank")
        self.represented_tp_ranks = represented
        if not isinstance(calibration_id, str):
            raise TypeError("calibration_id must be a string")
        self.calibration_id = calibration_id
        self.dense_prefix_layers = _strict_int(
            "dense_prefix_layers", dense_prefix_layers
        )
        if self.dense_prefix_layers > self.num_layers:
            raise ValueError("dense_prefix_layers cannot exceed num_layers")
        if self.num_heads % self.tp_size != 0:
            raise ValueError("num_heads must be divisible by tp_size")
        if self.num_output_groups % self.tp_size != 0:
            raise ValueError("num_output_groups must be divisible by tp_size")
        if self.num_heads % self.num_output_groups != 0:
            raise ValueError("num_heads must be divisible by num_output_groups")
        self.expected_segments = self._strict_names(
            "expected_segments", expected_segments
        )
        self.expected_oracles = self._strict_names("expected_oracles", expected_oracles)
        self._snapshots: Dict[Tuple[int, str], _ProjectionRows] = {}
        self._oracles: Dict[Tuple[int, str, str], _ProjectionRows] = {}
        self._projection_rank: Optional[int] = None
        self._sealed = False

    @staticmethod
    def _strict_names(name: str, values: Sequence[str]) -> Tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of names")
        result = tuple(values)
        if not result or any(not isinstance(x, str) or not x for x in result):
            raise ValueError(f"{name} must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} contains duplicates")
        return result

    def _validate_identity(self, layer_id: int, segment_id: str) -> int:
        layer = _strict_int("layer_id", layer_id)
        if layer >= self.num_layers:
            raise IndexError(f"layer_id={layer} is outside [0, {self.num_layers})")
        if segment_id not in self.expected_segments:
            raise ValueError(f"unexpected segment_id={segment_id!r}")
        return layer

    def _canonicalize(
        self,
        row_ids: torch.Tensor,
        projection: torch.Tensor,
    ) -> _ProjectionRows:
        if not torch.is_tensor(row_ids):
            row_ids = torch.as_tensor(row_ids)
        if row_ids.ndim != 1 or row_ids.device.type != "cpu":
            raise ValueError("row_ids must be a one-dimensional CPU tensor")
        if row_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("row_ids must use an integer dtype")
        if not torch.is_tensor(projection) or projection.ndim != 3:
            raise ValueError("projection must have shape [rows, heads, rank]")
        if projection.shape[0] != row_ids.numel():
            raise ValueError("projection row count does not match row_ids")
        if projection.shape[1] != self.num_heads:
            raise ValueError(
                f"projection heads={projection.shape[1]} != {self.num_heads}"
            )
        if projection.shape[0] <= 0 or projection.shape[2] <= 0:
            raise ValueError("projection row and rank dimensions must be positive")
        if not projection.dtype.is_floating_point:
            raise ValueError("projection must use a floating-point dtype")
        rank = int(projection.shape[2])
        if self._projection_rank is None:
            self._projection_rank = rank
        elif rank != self._projection_rank:
            raise ValueError(
                f"projection rank={rank} differs from {self._projection_rank}"
            )

        rows = row_ids.to(dtype=torch.int64).clone()
        order = torch.argsort(rows)
        rows = rows.index_select(0, order)
        if bool((rows < 0).any().item()):
            raise ValueError("row_ids must be non-negative")
        if rows.numel() > 1 and bool((rows[1:] == rows[:-1]).any().item()):
            raise ValueError("row_ids must be unique")
        values = (
            projection.detach().index_select(0, order.to(projection.device)).clone()
        )
        return _ProjectionRows(rows, values)

    def add_snapshot(
        self,
        layer_id: int,
        segment_id: str,
        row_ids: torch.Tensor,
        projection: torch.Tensor,
    ) -> None:
        if self._sealed:
            raise RuntimeError("collector is already finalized")
        layer = self._validate_identity(layer_id, segment_id)
        key = (layer, segment_id)
        if key in self._snapshots:
            raise ValueError(
                f"duplicate snapshot for layer={layer}, segment={segment_id}"
            )
        self._snapshots[key] = self._canonicalize(row_ids, projection)

    def add_oracle(
        self,
        layer_id: int,
        segment_id: str,
        oracle_id: str,
        row_ids: torch.Tensor,
        projection: torch.Tensor,
    ) -> None:
        if self._sealed:
            raise RuntimeError("collector is already finalized")
        layer = self._validate_identity(layer_id, segment_id)
        if oracle_id not in self.expected_oracles:
            raise ValueError(f"unexpected oracle_id={oracle_id!r}")
        key = (layer, segment_id, oracle_id)
        if key in self._oracles:
            raise ValueError(
                f"duplicate oracle for layer={layer}, segment={segment_id}, "
                f"oracle={oracle_id}"
            )
        self._oracles[key] = self._canonicalize(row_ids, projection)

    def _expected_snapshot_keys(self) -> set:
        return {
            (layer, segment)
            for layer in range(self.num_layers)
            for segment in self.expected_segments
        }

    def _expected_oracle_keys(self) -> set:
        return {
            (layer, segment, oracle)
            for layer in range(self.num_layers)
            for segment in self.expected_segments
            for oracle in self.expected_oracles
        }

    def finalize(self) -> MLAHeadDriftReport:
        if self._sealed:
            raise RuntimeError("collector is already finalized")
        missing_snapshots = self._expected_snapshot_keys() - set(self._snapshots)
        missing_oracles = self._expected_oracle_keys() - set(self._oracles)
        if missing_snapshots or missing_oracles:
            raise RuntimeError(
                "incomplete drift profile: "
                f"missing_snapshots={sorted(missing_snapshots)} "
                f"missing_oracles={sorted(missing_oracles)}"
            )

        heads_per_rank = self.num_heads // self.tp_size
        groups_per_rank = self.num_output_groups // self.tp_size
        heads_per_group = self.num_heads // self.num_output_groups
        pair_reports: List[DriftPairMetrics] = []
        aggregate_by_layer: List[Tuple[HeadDriftMetrics, ...]] = []
        rank_gram = None
        rank_reference = None
        pair_rank_grams: List[torch.Tensor] = []
        pair_rank_references: List[torch.Tensor] = []
        layer_snapshots: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        layer_oracles: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        layer_pair_metrics: List[List[Tuple[HeadDriftMetrics, ...]]] = [
            [] for _ in range(self.num_layers)
        ]

        for layer in range(self.num_layers):
            for segment in self.expected_segments:
                snapshot = self._snapshots[(layer, segment)]
                for oracle_id in self.expected_oracles:
                    oracle = self._oracles[(layer, segment, oracle_id)]
                    if not torch.equal(snapshot.row_ids, oracle.row_ids):
                        raise RuntimeError(
                            "snapshot/oracle row mismatch at "
                            f"layer={layer}, segment={segment}, oracle={oracle_id}"
                        )
                    if snapshot.values.shape != oracle.values.shape:
                        raise RuntimeError(
                            "snapshot/oracle projection shape mismatch at "
                            f"layer={layer}, segment={segment}, oracle={oracle_id}"
                        )
                    if snapshot.values.device != oracle.values.device:
                        raise RuntimeError(
                            "snapshot/oracle projections must share a device at "
                            f"layer={layer}, segment={segment}, oracle={oracle_id}"
                        )
                    if snapshot.values.dtype != oracle.values.dtype:
                        raise RuntimeError(
                            "snapshot/oracle projection dtypes must match at "
                            f"layer={layer}, segment={segment}, oracle={oracle_id}"
                        )
                    if not bool(
                        torch.isfinite(snapshot.values).all().item()
                    ) or not bool(torch.isfinite(oracle.values).all().item()):
                        raise RuntimeError(
                            "non-finite projection at "
                            f"layer={layer}, segment={segment}, oracle={oracle_id}"
                        )

                    snap = snapshot.values.float()
                    ref = oracle.values.float()
                    metrics = _metrics_from_tensors(snap, ref)
                    pair_reports.append(
                        DriftPairMetrics(
                            layer_id=layer,
                            segment_id=segment,
                            oracle_id=oracle_id,
                            row_count=int(snapshot.row_ids.numel()),
                            row_ids_sha256=hashlib.sha256(
                                ",".join(
                                    str(value) for value in snapshot.row_ids.tolist()
                                ).encode("ascii")
                            ).hexdigest(),
                            head_metrics=metrics,
                        )
                    )
                    layer_snapshots[layer].append(snap)
                    layer_oracles[layer].append(ref)
                    layer_pair_metrics[layer].append(metrics)

                    if rank_gram is None:
                        device = snap.device
                        rank_gram = torch.zeros(
                            self.num_layers,
                            self.tp_size,
                            heads_per_rank,
                            heads_per_rank,
                            dtype=torch.float64,
                            device=device,
                        )
                        rank_reference = torch.zeros(
                            self.num_layers,
                            self.tp_size,
                            dtype=torch.float64,
                            device=device,
                        )
                    elif rank_gram.device != snap.device:
                        raise RuntimeError("all profile pairs must share one device")

                    delta = (snap - ref).double()
                    ref64 = ref.double()
                    this_pair_gram = torch.zeros(
                        self.tp_size,
                        heads_per_rank,
                        heads_per_rank,
                        dtype=torch.float64,
                        device=delta.device,
                    )
                    this_pair_reference = torch.zeros(
                        self.tp_size,
                        dtype=torch.float64,
                        device=delta.device,
                    )
                    for rank_id in range(self.tp_size):
                        start = rank_id * heads_per_rank
                        end = start + heads_per_rank
                        # Different wo_a output groups occupy different G
                        # coordinates.  They are orthogonal and must never be
                        # added in the shared R coordinate, where artificial
                        # cross-group cancellation would corrupt the selector.
                        rank_delta = delta[:, start:end, :].reshape(
                            delta.shape[0],
                            groups_per_rank,
                            heads_per_group,
                            delta.shape[-1],
                        )
                        rank_ref = ref64[:, start:end, :].reshape(
                            ref64.shape[0],
                            groups_per_rank,
                            heads_per_group,
                            ref64.shape[-1],
                        )
                        for group_id in range(groups_per_rank):
                            group_start = group_id * heads_per_group
                            group_end = group_start + heads_per_group
                            flat_delta = (
                                rank_delta[:, group_id]
                                .permute(1, 0, 2)
                                .reshape(heads_per_group, -1)
                            )
                            this_pair_gram[
                                rank_id,
                                group_start:group_end,
                                group_start:group_end,
                            ].add_(flat_delta @ flat_delta.transpose(0, 1))
                        summed_ref = rank_ref.sum(dim=2)
                        this_pair_reference[rank_id].add_(summed_ref.square().sum())
                    rank_gram[layer].add_(this_pair_gram)
                    rank_reference[layer].add_(this_pair_reference)
                    pair_rank_grams.append(this_pair_gram)
                    pair_rank_references.append(this_pair_reference)

        if rank_gram is None or rank_reference is None:
            raise RuntimeError("profile contains no paired observations")
        for layer in range(self.num_layers):
            aggregate_by_layer.append(
                _aggregate_metrics(
                    layer_snapshots[layer],
                    layer_oracles[layer],
                    layer_pair_metrics[layer],
                )
            )

        self._sealed = True
        return MLAHeadDriftReport(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_output_groups=self.num_output_groups,
            tp_size=self.tp_size,
            tp_world_size=self.tp_world_size,
            represented_tp_ranks=self.represented_tp_ranks,
            calibration_id=self.calibration_id,
            dense_prefix_layers=self.dense_prefix_layers,
            expected_segments=self.expected_segments,
            expected_oracles=self.expected_oracles,
            aggregate_metrics=tuple(aggregate_by_layer),
            pair_metrics=tuple(pair_reports),
            rank_gram=rank_gram,
            rank_reference_energy=rank_reference,
            pair_rank_gram=torch.stack(pair_rank_grams, dim=0),
            pair_rank_reference_energy=torch.stack(pair_rank_references, dim=0),
        )


def merge_tp_drift_reports(
    shards: Sequence[MLAHeadDriftReport],
) -> MLAHeadDriftReport:
    """Merge identified single-rank reports in contiguous TP head order.

    Input sequence order is deliberately ignored.  Every shard must carry one
    unique ``represented_tp_ranks`` identity and the complete set
    ``0..tp_world_size-1`` must be present, preventing a filename/list ordering
    mistake from silently assigning metrics to the wrong logical heads.
    """

    shards = tuple(shards)
    if not shards:
        raise ValueError("at least one TP drift report is required")
    first = shards[0]
    world_size = first.tp_world_size
    by_rank = {}
    for shard in shards:
        if shard.tp_size != 1 or len(shard.represented_tp_ranks) != 1:
            raise ValueError("every TP drift shard must represent exactly one rank")
        if shard.tp_world_size != world_size:
            raise ValueError("TP drift shard world-size mismatch")
        rank = shard.represented_tp_ranks[0]
        if rank in by_rank:
            raise ValueError(f"duplicate TP drift shard rank {rank}")
        by_rank[rank] = shard
    expected_ranks = set(range(world_size))
    if set(by_rank) != expected_ranks:
        raise ValueError(
            "TP drift shards are incomplete: "
            f"expected={sorted(expected_ranks)} observed={sorted(by_rank)}"
        )
    shards = tuple(by_rank[rank] for rank in range(world_size))
    first = shards[0]
    local_heads = first.num_heads
    local_groups = first.num_output_groups
    pair_keys = tuple(
        (
            x.layer_id,
            x.segment_id,
            x.oracle_id,
            x.row_count,
            x.row_ids_sha256,
        )
        for x in first.pair_metrics
    )
    for rank, shard in enumerate(shards):
        if (
            shard.num_layers != first.num_layers
            or shard.num_heads != local_heads
            or shard.num_output_groups != local_groups
            or shard.dense_prefix_layers != first.dense_prefix_layers
            or shard.expected_segments != first.expected_segments
            or shard.expected_oracles != first.expected_oracles
            or shard.calibration_id != first.calibration_id
        ):
            raise ValueError(f"TP drift shard {rank} metadata mismatch")
        keys = tuple(
            (
                x.layer_id,
                x.segment_id,
                x.oracle_id,
                x.row_count,
                x.row_ids_sha256,
            )
            for x in shard.pair_metrics
        )
        if keys != pair_keys:
            raise ValueError(f"TP drift shard {rank} pair ordering mismatch")
        if shard.rank_gram.device != first.rank_gram.device:
            raise ValueError("TP drift reports must share a device")

    aggregate = []
    for layer in range(first.num_layers):
        aggregate.append(
            tuple(
                metric for shard in shards for metric in shard.aggregate_metrics[layer]
            )
        )
    pairs = []
    for pair_index, key in enumerate(pair_keys):
        pairs.append(
            DriftPairMetrics(
                layer_id=key[0],
                segment_id=key[1],
                oracle_id=key[2],
                row_count=key[3],
                row_ids_sha256=key[4],
                head_metrics=tuple(
                    metric
                    for shard in shards
                    for metric in shard.pair_metrics[pair_index].head_metrics
                ),
            )
        )
    gram = torch.cat(tuple(x.rank_gram for x in shards), dim=1)
    reference = torch.cat(tuple(x.rank_reference_energy for x in shards), dim=1)
    pair_gram = torch.cat(tuple(x.pair_rank_gram for x in shards), dim=1)
    pair_reference = torch.cat(
        tuple(x.pair_rank_reference_energy for x in shards), dim=1
    )
    return MLAHeadDriftReport(
        num_layers=first.num_layers,
        num_heads=local_heads * len(shards),
        num_output_groups=local_groups * len(shards),
        tp_size=len(shards),
        tp_world_size=world_size,
        represented_tp_ranks=tuple(range(world_size)),
        calibration_id=first.calibration_id,
        dense_prefix_layers=first.dense_prefix_layers,
        expected_segments=first.expected_segments,
        expected_oracles=first.expected_oracles,
        aggregate_metrics=tuple(aggregate),
        pair_metrics=tuple(pairs),
        rank_gram=gram,
        rank_reference_energy=reference,
        pair_rank_gram=pair_gram,
        pair_rank_reference_energy=pair_reference,
    )


@dataclass(frozen=True)
class HeadSelectionThresholds:
    """Explicit fail-closed thresholds for local-head admission."""

    max_head_relative_rms: float
    max_head_row_p99: float
    max_head_row_max: float
    min_head_cosine: float
    max_rank_no_cancel_error: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_head_relative_rms",
            _finite_nonnegative("max_head_relative_rms", self.max_head_relative_rms),
        )
        object.__setattr__(
            self,
            "max_head_row_p99",
            _finite_nonnegative("max_head_row_p99", self.max_head_row_p99),
        )
        object.__setattr__(
            self,
            "max_head_row_max",
            _finite_nonnegative("max_head_row_max", self.max_head_row_max),
        )
        cosine = float(self.min_head_cosine)
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            raise ValueError("min_head_cosine must be finite and in [-1, 1]")
        object.__setattr__(self, "min_head_cosine", cosine)
        object.__setattr__(
            self,
            "max_rank_no_cancel_error",
            _finite_nonnegative(
                "max_rank_no_cancel_error", self.max_rank_no_cancel_error
            ),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_head_relative_rms": self.max_head_relative_rms,
            "max_head_row_p99": self.max_head_row_p99,
            "max_head_row_max": self.max_head_row_max,
            "min_head_cosine": self.min_head_cosine,
            "max_rank_no_cancel_error": self.max_rank_no_cancel_error,
        }


@dataclass(frozen=True)
class TPBalancedHeadSelection:
    """One member of a nested, TP-balanced local-head policy family."""

    num_layers: int
    num_heads: int
    tp_size: int
    dense_prefix_layers: int
    target_local_heads_per_rank: int
    local_heads_by_layer: Tuple[Tuple[int, ...], ...]
    local_heads_per_rank_by_layer: Tuple[int, ...]
    conservative_rank_error_by_layer: Tuple[Tuple[float, ...], ...]
    thresholds: HeadSelectionThresholds
    selection_mode: str = STRICT_SELECTION_MODE

    def __post_init__(self) -> None:
        if self.selection_mode not in {
            STRICT_SELECTION_MODE,
            EXPLORATORY_FIXED_COUNT_SELECTION_MODE,
        }:
            raise ValueError(f"unsupported selection_mode={self.selection_mode!r}")

    @property
    def realized_local_heads(self) -> int:
        return sum(len(row) for row in self.local_heads_by_layer)

    @property
    def eligible_non_dense_heads(self) -> int:
        return (self.num_layers - self.dense_prefix_layers) * self.num_heads

    @property
    def realized_local_ratio(self) -> float:
        denom = self.eligible_non_dense_heads
        return self.realized_local_heads / denom if denom > 0 else 0.0

    @property
    def whole_model_local_ratio(self) -> float:
        denom = self.num_layers * self.num_heads
        return self.realized_local_heads / denom if denom > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "tp_size": self.tp_size,
            "dense_prefix_layers": self.dense_prefix_layers,
            "target_local_heads_per_rank": self.target_local_heads_per_rank,
            "local_heads_by_layer": [list(x) for x in self.local_heads_by_layer],
            "local_heads_per_rank_by_layer": list(self.local_heads_per_rank_by_layer),
            "conservative_rank_error_by_layer": [
                list(x) for x in self.conservative_rank_error_by_layer
            ],
            "realized_local_heads": self.realized_local_heads,
            "realized_local_ratio": self.realized_local_ratio,
            "whole_model_local_ratio": self.whole_model_local_ratio,
            "thresholds": self.thresholds.to_dict(),
            "selection_mode": self.selection_mode,
            "threshold_admission_bypassed": (
                self.selection_mode == EXPLORATORY_FIXED_COUNT_SELECTION_MODE
            ),
            "safety": (
                "rank_cost_is_worst_pair_sum_of_head_norms_no_cancellation"
                if self.selection_mode == STRICT_SELECTION_MODE
                else "unqualified_exploratory_threshold_bypass_requires_gate1"
            ),
        }


def _head_is_eligible(
    metrics: HeadDriftMetrics,
    no_cancel_cost: float,
    thresholds: HeadSelectionThresholds,
) -> bool:
    values = (
        metrics.worst_pair_relative_rms,
        metrics.worst_pair_row_p99,
        metrics.worst_pair_row_max,
        metrics.worst_pair_cosine,
        no_cancel_cost,
    )
    if not all(math.isfinite(x) for x in values):
        return False
    return (
        metrics.worst_pair_relative_rms <= thresholds.max_head_relative_rms
        and metrics.worst_pair_row_p99 <= thresholds.max_head_row_p99
        and metrics.worst_pair_row_max <= thresholds.max_head_row_max
        and metrics.worst_pair_cosine >= thresholds.min_head_cosine
        and no_cancel_cost <= thresholds.max_rank_no_cancel_error
    )


def _head_risk_key(
    metrics: HeadDriftMetrics,
    no_cancel_cost: float,
    head_id: int,
) -> Tuple[float, float, float, float, float, int]:
    """Return the selector's stable, deterministic per-head risk ordering."""

    return (
        float(no_cancel_cost),
        float(metrics.worst_pair_relative_rms),
        float(metrics.worst_pair_row_p99),
        float(metrics.worst_pair_row_max),
        -float(metrics.worst_pair_cosine),
        int(head_id),
    )


def _validate_selection_inputs(
    report: MLAHeadDriftReport,
    target_local_heads_per_rank: Sequence[int],
) -> Tuple[int, ...]:
    if not isinstance(report, MLAHeadDriftReport):
        raise TypeError("report must be an MLAHeadDriftReport")
    targets = tuple(target_local_heads_per_rank)
    if not targets:
        raise ValueError("target_local_heads_per_rank cannot be empty")
    for value in targets:
        _strict_int("target_local_heads_per_rank", value, minimum=1)
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target local-head counts must be strictly increasing")
    if targets[-1] > report.heads_per_rank:
        raise ValueError(
            f"target={targets[-1]} exceeds heads_per_rank={report.heads_per_rank}"
        )
    return targets


def _assert_nested_selections(
    selections: Sequence[TPBalancedHeadSelection],
) -> None:
    for previous, current in zip(selections, selections[1:]):
        for layer in range(previous.num_layers):
            if not set(previous.local_heads_by_layer[layer]).issubset(
                current.local_heads_by_layer[layer]
            ):
                raise RuntimeError("selector violated nested-policy invariant")


def select_tp_balanced_nested(
    report: MLAHeadDriftReport,
    *,
    thresholds: HeadSelectionThresholds,
    target_local_heads_per_rank: Sequence[int],
) -> Tuple[TPBalancedHeadSelection, ...]:
    """Build nested TP-balanced policies from fixed per-rank head orderings.

    Heads are sorted once per ``(layer, TP rank)`` by their individual
    no-cancellation cost and worst-pair drift.  Every requested target then
    takes a prefix of that fixed order.  A layer uses the minimum safe prefix
    length across all ranks, guaranteeing equal local-head counts on every TP
    shard.  Increasing targets can therefore only add heads; it cannot replace
    a previously selected head.
    """

    targets = _validate_selection_inputs(report, target_local_heads_per_rank)

    # Fixed order and maximum conservatively safe prefix for every rank/layer.
    orders: List[List[Tuple[int, ...]]] = []
    safe_prefixes: List[List[int]] = []
    prefix_costs: List[List[Tuple[float, ...]]] = []
    for layer in range(report.num_layers):
        layer_orders = []
        layer_safe = []
        layer_costs = []
        for rank in range(report.tp_size):
            start = rank * report.heads_per_rank
            candidates = []
            for head in range(start, start + report.heads_per_rank):
                metrics = report.aggregate_metrics[layer][head]
                cost = report.head_no_cancel_cost(layer, head)
                if _head_is_eligible(metrics, cost, thresholds):
                    candidates.append(_head_risk_key(metrics, cost, head))
            candidates.sort()
            order = tuple(int(x[-1]) for x in candidates)
            costs = tuple(float(x[0]) for x in candidates)
            cumulative = 0.0
            safe = 0
            cumulative_costs = []
            for cost in costs:
                cumulative += cost
                cumulative_costs.append(cumulative)
                if cumulative <= thresholds.max_rank_no_cancel_error:
                    safe += 1
                else:
                    break
            layer_orders.append(order)
            layer_safe.append(safe)
            layer_costs.append(tuple(cumulative_costs))
        orders.append(layer_orders)
        safe_prefixes.append(layer_safe)
        prefix_costs.append(layer_costs)

    selections = []
    for target in targets:
        local_by_layer = []
        count_by_layer = []
        errors_by_layer = []
        for layer in range(report.num_layers):
            if layer < report.dense_prefix_layers:
                local_by_layer.append(())
                count_by_layer.append(0)
                errors_by_layer.append(tuple(0.0 for _ in range(report.tp_size)))
                continue
            balanced_count = min(
                target,
                min(safe_prefixes[layer]),
            )
            selected = tuple(
                head
                for rank in range(report.tp_size)
                for head in orders[layer][rank][:balanced_count]
            )
            rank_errors = tuple(
                (
                    prefix_costs[layer][rank][balanced_count - 1]
                    if balanced_count > 0
                    else 0.0
                )
                for rank in range(report.tp_size)
            )
            local_by_layer.append(tuple(sorted(selected)))
            count_by_layer.append(balanced_count)
            errors_by_layer.append(rank_errors)
        selections.append(
            TPBalancedHeadSelection(
                num_layers=report.num_layers,
                num_heads=report.num_heads,
                tp_size=report.tp_size,
                dense_prefix_layers=report.dense_prefix_layers,
                target_local_heads_per_rank=target,
                local_heads_by_layer=tuple(local_by_layer),
                local_heads_per_rank_by_layer=tuple(count_by_layer),
                conservative_rank_error_by_layer=tuple(errors_by_layer),
                thresholds=thresholds,
            )
        )

    # Defensive invariant check: later policies may only add heads.
    _assert_nested_selections(selections)
    return tuple(selections)


def select_tp_balanced_fixed_count_exploratory(
    report: MLAHeadDriftReport,
    *,
    thresholds: HeadSelectionThresholds,
    target_local_heads_per_rank: Sequence[int],
) -> Tuple[TPBalancedHeadSelection, ...]:
    """Build explicit unqualified fixed-count candidates ranked by drift.

    Unlike :func:`select_tp_balanced_nested`, this function deliberately does
    not use the individual-head thresholds or the cumulative no-cancellation
    budget as admission gates.  It sorts every finite-evidence head with the
    exact same risk tuple as the strict selector, then takes a fixed prefix per
    ``(layer, TP rank)``.  The result is useful only as a clearly labelled
    exploratory input to the held-out Gate1 comparison; it is not an accuracy
    or performance qualification.

    Non-finite evidence is rejected instead of being sorted last and selected
    merely to satisfy the requested count.
    """

    targets = _validate_selection_inputs(report, target_local_heads_per_rank)
    if not isinstance(thresholds, HeadSelectionThresholds):
        raise TypeError("thresholds must be HeadSelectionThresholds")

    orders: List[List[Tuple[int, ...]]] = []
    prefix_costs: List[List[Tuple[float, ...]]] = []
    for layer in range(report.num_layers):
        layer_orders = []
        layer_costs = []
        for rank in range(report.tp_size):
            start = rank * report.heads_per_rank
            candidates = []
            for head in range(start, start + report.heads_per_rank):
                metrics = report.aggregate_metrics[layer][head]
                cost = report.head_no_cancel_cost(layer, head)
                risk = _head_risk_key(metrics, cost, head)
                if not all(math.isfinite(value) for value in risk[:-1]):
                    raise ValueError(
                        "exploratory fixed-count selection refuses non-finite "
                        f"drift evidence at layer={layer} rank={rank} head={head}"
                    )
                candidates.append(risk)
            candidates.sort()
            layer_orders.append(tuple(int(item[-1]) for item in candidates))
            cumulative = 0.0
            cumulative_costs = []
            for item in candidates:
                cumulative += float(item[0])
                cumulative_costs.append(cumulative)
            layer_costs.append(tuple(cumulative_costs))
        orders.append(layer_orders)
        prefix_costs.append(layer_costs)

    selections = []
    for target in targets:
        local_by_layer = []
        count_by_layer = []
        errors_by_layer = []
        for layer in range(report.num_layers):
            if layer < report.dense_prefix_layers:
                local_by_layer.append(())
                count_by_layer.append(0)
                errors_by_layer.append(tuple(0.0 for _ in range(report.tp_size)))
                continue
            selected = tuple(
                head
                for rank in range(report.tp_size)
                for head in orders[layer][rank][:target]
            )
            rank_errors = tuple(
                prefix_costs[layer][rank][target - 1]
                for rank in range(report.tp_size)
            )
            local_by_layer.append(tuple(sorted(selected)))
            count_by_layer.append(target)
            errors_by_layer.append(rank_errors)
        selections.append(
            TPBalancedHeadSelection(
                num_layers=report.num_layers,
                num_heads=report.num_heads,
                tp_size=report.tp_size,
                dense_prefix_layers=report.dense_prefix_layers,
                target_local_heads_per_rank=target,
                local_heads_by_layer=tuple(local_by_layer),
                local_heads_per_rank_by_layer=tuple(count_by_layer),
                conservative_rank_error_by_layer=tuple(errors_by_layer),
                thresholds=thresholds,
                selection_mode=EXPLORATORY_FIXED_COUNT_SELECTION_MODE,
            )
        )

    _assert_nested_selections(selections)
    return tuple(selections)


def build_deepseek_v4_head_config_dict(
    selection: TPBalancedHeadSelection,
    *,
    local_window: int = 128,
    swa_capacity: int = 128,
    sink_size: int = 4,
    profiling_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert one balanced selection into a loadable DeepSeek V4 config."""

    if not isinstance(selection, TPBalancedHeadSelection):
        raise TypeError("selection must be a TPBalancedHeadSelection")
    window = _strict_int("local_window", local_window, minimum=1)
    capacity = _strict_int("swa_capacity", swa_capacity, minimum=1)
    sink = _strict_int("sink_size", sink_size, minimum=0)
    if window > capacity:
        raise ValueError(
            f"local_window={window} exceeds executable SWA capacity={capacity}"
        )
    if len(selection.local_heads_by_layer) != selection.num_layers:
        raise ValueError("selection layer count is inconsistent")
    if selection.num_heads % selection.tp_size != 0:
        raise ValueError("selection num_heads must be divisible by tp_size")
    heads_per_rank = selection.num_heads // selection.tp_size

    head_class = []
    head_distance = []
    head_sinks = []
    for layer in range(selection.num_layers):
        selected = tuple(selection.local_heads_by_layer[layer])
        if len(set(selected)) != len(selected) or any(
            isinstance(x, bool) or not isinstance(x, int) for x in selected
        ):
            raise ValueError(f"layer {layer} has invalid/duplicate local head IDs")
        if any(x < 0 or x >= selection.num_heads for x in selected):
            raise ValueError(f"layer {layer} has an out-of-range local head")
        if layer < selection.dense_prefix_layers and selected:
            raise ValueError(f"dense-prefix layer {layer} cannot contain local heads")
        rank_counts = [0] * selection.tp_size
        for head in selected:
            rank_counts[head // heads_per_rank] += 1
        if selected and len(set(rank_counts)) != 1:
            raise ValueError(
                f"layer {layer} is TP-asymmetric: rank local counts={rank_counts}"
            )
        expected_count = selection.local_heads_per_rank_by_layer[layer]
        if any(count != expected_count for count in rank_counts):
            raise ValueError(
                f"layer {layer} local count differs from selection metadata: "
                f"counts={rank_counts}, expected={expected_count}"
            )

        selected_set = set(selected)
        if layer < selection.dense_prefix_layers:
            classes = ["dense"] * selection.num_heads
            distances = [-1] * selection.num_heads
        else:
            classes = [
                "local" if head in selected_set else "global"
                for head in range(selection.num_heads)
            ]
            distances = [
                window if head in selected_set else -1
                for head in range(selection.num_heads)
            ]
        head_class.append(classes)
        head_distance.append(distances)
        head_sinks.append([sink] * selection.num_heads)

    exploratory = (
        selection.selection_mode == EXPLORATORY_FIXED_COUNT_SELECTION_MODE
    )
    meta = {
        "selector": (
            "tp_balanced_fixed_count_projected_drift_exploratory_v1"
            if exploratory
            else "tp_balanced_nested_projected_drift_v1"
        ),
        "safety": (
            "unqualified_exploratory_threshold_bypass_requires_gate1"
            if exploratory
            else "worst_pair_additive_head_norm_no_error_cancellation"
        ),
        "requires_reuse_heads_full_scope": True,
        "target_local_heads_per_rank": selection.target_local_heads_per_rank,
        "realized_local_heads": selection.realized_local_heads,
        "realized_local_ratio": selection.realized_local_ratio,
        "whole_model_local_ratio": selection.whole_model_local_ratio,
        "thresholds": selection.thresholds.to_dict(),
        "swa_capacity": capacity,
    }
    if exploratory:
        meta.update(
            {
                "candidate_tier": "exploratory_fixed_count",
                "selection_mode": EXPLORATORY_FIXED_COUNT_SELECTION_MODE,
                "threshold_admission_bypassed": True,
                "strict_threshold_qualified": False,
                "screening_only": True,
                "accuracy_claim_qualified": False,
                "performance_claim_qualified": False,
                "gate1_required_before_offline_reuse": True,
            }
        )
    if profiling_meta is not None:
        if not isinstance(profiling_meta, Mapping):
            raise TypeError("profiling_meta must be a mapping")
        conflicts = sorted(set(meta).intersection(profiling_meta))
        if conflicts:
            raise ValueError(
                "profiling_meta cannot override reserved keys: " + ", ".join(conflicts)
            )
        meta.update(dict(profiling_meta))

    return {
        "format": HEAD_CONFIG_FORMAT,
        "num_layers": selection.num_layers,
        "num_attention_heads": selection.num_heads,
        "physical_kv_heads": 1,
        "dense_prefix_layers": selection.dense_prefix_layers,
        "default_sink_size": sink,
        "local_default_window": window,
        "mla_head_classification": head_class,
        "mla_head_max_distance": head_distance,
        "mla_head_sink_size": head_sinks,
        "profiling_meta": meta,
    }


__all__ = [
    "DRIFT_REPORT_FORMAT",
    "EXPLORATORY_FIXED_COUNT_SELECTION_MODE",
    "HEAD_CONFIG_FORMAT",
    "STRICT_SELECTION_MODE",
    "DriftPairMetrics",
    "HeadDriftMetrics",
    "HeadSelectionThresholds",
    "MLAHeadDriftCollector",
    "MLAHeadDriftReport",
    "TPBalancedHeadSelection",
    "build_deepseek_v4_head_config_dict",
    "decompose_wo_a_per_head",
    "merge_tp_drift_reports",
    "select_tp_balanced_fixed_count_exploratory",
    "select_tp_balanced_nested",
    "sum_wo_a_head_contributions",
]
