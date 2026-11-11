"""CPU preflight for sparse-anomaly probe budgets, not runtime authorization.

The finite-population result applies ONLY to fresh uniform sampling without
replacement and anomalies fixed before the draw. It does not apply to the
existing deterministic stratified gate, adversarial post-draw changes, numerical
false negatives, or EM/F1. A small probe budget is not evidence of safety.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import comb, isfinite

from .core import integer, require

MAX_CANDIDATE_ROWS = 65536
UNIFORM = "fresh_uniform_without_replacement"
DETERMINISTIC = "deterministic_stratified"


def _population(rows, bad_rows, probes):
    for value, name in ((rows, "candidate_rows"), (bad_rows, "min_bad_rows"),
                        (probes, "probes")):
        integer(value, name)
    require(rows <= MAX_CANDIDATE_ROWS, "population exceeds bounded CPU planner")
    require(1 <= bad_rows <= rows, "min_bad_rows must be in [1, candidate_rows]")
    require(probes <= rows, "probes exceed population")


def miss_probability(candidate_rows, min_bad_rows, probes):
    """Exact Fraction: P(no bad row drawn | at least min_bad_rows bad).

    Conditional upper bound for uniform probes, NOT posterior probability that
    a passing block is safe. An exact rational avoids rounding tiny risks to 0.
    The population excludes rows already recomputed (guards/previous probes).
    """
    _population(candidate_rows, min_bad_rows, probes)
    if probes > candidate_rows - min_bad_rows:
        return Fraction(0)
    # Symmetric form is substantially cheaper for sparse anomalies.
    if min_bad_rows < probes:
        return Fraction(comb(candidate_rows - probes, min_bad_rows),
                        comb(candidate_rows, min_bad_rows))
    return Fraction(comb(candidate_rows - min_bad_rows, probes),
                    comb(candidate_rows, probes))


def _risk(value):
    require(type(value) in (float, int), "explicit real max_miss_probability required")
    require(0 <= value < 1 and isfinite(value), "risk must be finite in [0, 1)")
    return Fraction(value)


def required_probes(candidate_rows, *, min_bad_rows=1, max_miss_probability):
    """Smallest uniform probe count meeting an explicit conditional risk cap."""
    _population(candidate_rows, min_bad_rows, 0)
    risk = _risk(max_miss_probability)
    low, high = 0, candidate_rows - min_bad_rows + 1
    while low < high:
        middle = (low + high) // 2
        if miss_probability(candidate_rows, min_bad_rows, middle) <= risk:
            high = middle
        else:
            low = middle + 1
    return low


@dataclass(frozen=True)
class CoverageBudget:
    candidate_rows: int
    min_bad_rows: int
    available_probes: int
    required_uniform_probes: int
    max_miss_probability: float
    sampling_scheme: str
    recommendation: str
    reason: str

    def to_dict(self):
        return dict(asdict(self), runtime_admission_created=False,
            conditional_bound_not_accuracy_guarantee=True,
            scope="one block; not a request-wide probability budget")


def plan_coverage(candidate_rows, *, available_probes, max_miss_probability,
                  min_bad_rows=1, sampling_scheme=DETERMINISTIC):
    """Fail-closed budget preflight; callers still need numerical validation.

    Defaults to the current deterministic schedule: its coverage cannot be
    justified with a random-sampling bound. Insufficient budget or full coverage
    requirement recommends recomputation, never lowering the requested standard.
    Even 'collect_uniform_probes' is NOT an accept/reuse decision.
    """
    _population(candidate_rows, min_bad_rows, available_probes)
    require(sampling_scheme in (UNIFORM, DETERMINISTIC), "unknown sampling scheme")
    count = required_probes(candidate_rows, min_bad_rows=min_bad_rows,
                            max_miss_probability=max_miss_probability)
    if sampling_scheme != UNIFORM:
        recommendation, reason = "full_recompute", "random_coverage_bound_not_applicable"
    elif count == candidate_rows:
        recommendation, reason = "full_recompute", "no_unprobed_rows_would_remain"
    elif count > available_probes:
        recommendation, reason = "full_recompute", "probe_budget_insufficient"
    else:
        recommendation, reason = "collect_uniform_probes", "budget_sufficient_not_yet_verified"
    return CoverageBudget(candidate_rows, min_bad_rows, available_probes, count,
                          float(max_miss_probability), sampling_scheme,
                          recommendation, reason)
