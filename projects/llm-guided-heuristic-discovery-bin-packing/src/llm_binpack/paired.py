"""Paired per-instance statistical comparison utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean, median

from .evaluator import Evaluation


@dataclass(frozen=True, slots=True)
class PairedComparison:
    n: int
    candidate_mean_bins: float
    baseline_mean_bins: float
    mean_delta_bins: float
    median_delta_bins: float
    wins: int
    ties: int
    losses: int
    exact_sign_pvalue: float
    bootstrap_mean_delta_ci95: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "candidate_mean_bins": self.candidate_mean_bins,
            "baseline_mean_bins": self.baseline_mean_bins,
            "mean_delta_bins": self.mean_delta_bins,
            "median_delta_bins": self.median_delta_bins,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "exact_sign_pvalue": self.exact_sign_pvalue,
            "bootstrap_mean_delta_ci95": list(self.bootstrap_mean_delta_ci95),
        }


def _exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    """Exact two-sided sign test, excluding ties."""

    n = wins + losses
    if n == 0:
        return 1.0

    k = min(wins, losses)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap_ci(
    deltas: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = random.Random(seed)
    n = len(deltas)
    means = [
        fmean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    ]
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def compare_to_best_fit(
    evaluation: Evaluation,
    *,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 73,
) -> PairedComparison:
    """Compare candidate and Best Fit on exactly the same ordered instances.

    Delta is candidate bins minus Best Fit bins, so negative is better.
    """

    if not evaluation.cases:
        raise ValueError("evaluation contains no cases")

    candidate = tuple(float(row.bins_used) for row in evaluation.cases)
    baseline = tuple(float(row.best_fit_bins) for row in evaluation.cases)
    deltas = tuple(c - b for c, b in zip(candidate, baseline, strict=True))

    wins = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta > 0 for delta in deltas)

    return PairedComparison(
        n=len(deltas),
        candidate_mean_bins=fmean(candidate),
        baseline_mean_bins=fmean(baseline),
        mean_delta_bins=fmean(deltas),
        median_delta_bins=float(median(deltas)),
        wins=wins,
        ties=ties,
        losses=losses,
        exact_sign_pvalue=_exact_two_sided_sign_pvalue(wins, losses),
        bootstrap_mean_delta_ci95=_paired_bootstrap_ci(
            deltas,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
    )
