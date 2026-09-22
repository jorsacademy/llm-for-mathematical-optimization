"""Evaluation of candidate heuristics against benchmark instances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from .baselines import best_fit
from .benchmarks import BenchmarkCase
from .expressions import compile_expression, expression_complexity
from .problem import pack_with_score


@dataclass(frozen=True, slots=True)
class CaseResult:
    family: str
    name: str
    source: str
    bins_used: int
    lower_bound: int
    best_fit_bins: int
    reference_optimum: int | None = None

    @property
    def excess_bins(self) -> int:
        return self.bins_used - self.lower_bound

    @property
    def excess_ratio(self) -> float:
        return self.excess_bins / max(1, self.lower_bound)

    @property
    def reference_gap(self) -> int | None:
        if self.reference_optimum is None:
            return None
        return self.bins_used - self.reference_optimum


@dataclass(frozen=True, slots=True)
class Evaluation:
    expression: str
    cases: tuple[CaseResult, ...]
    mean_bins: float
    mean_excess: float
    mean_excess_ratio: float
    best_fit_wins: int
    best_fit_ties: int
    best_fit_losses: int
    complexity: int
    fitness: float
    mean_reference_gap: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "mean_bins": self.mean_bins,
            "mean_excess": self.mean_excess,
            "mean_excess_ratio": self.mean_excess_ratio,
            "best_fit_wins": self.best_fit_wins,
            "best_fit_ties": self.best_fit_ties,
            "best_fit_losses": self.best_fit_losses,
            "complexity": self.complexity,
            "fitness": self.fitness,
            "mean_reference_gap": self.mean_reference_gap,
        }


def volume_lower_bound(case: BenchmarkCase) -> int:
    total = math.fsum(case.instance.items)
    return math.ceil((total / case.instance.capacity) - 1e-12)


def evaluate_expression(
    expression: str,
    suite: tuple[BenchmarkCase, ...],
    *,
    complexity_penalty: float = 1e-5,
) -> Evaluation:
    """Evaluate one expression. Higher fitness is better."""

    heuristic = compile_expression(expression)
    rows: list[CaseResult] = []

    for case in suite:
        candidate = pack_with_score(case.instance, heuristic)
        baseline = best_fit(case.instance)
        rows.append(
            CaseResult(
                family=case.family,
                name=case.instance.name,
                source=case.source,
                bins_used=candidate.bins_used,
                lower_bound=volume_lower_bound(case),
                best_fit_bins=baseline.bins_used,
                reference_optimum=case.reference_optimum,
            )
        )

    wins = sum(row.bins_used < row.best_fit_bins for row in rows)
    ties = sum(row.bins_used == row.best_fit_bins for row in rows)
    losses = len(rows) - wins - ties
    complexity = expression_complexity(expression)
    mean_excess_ratio = fmean(row.excess_ratio for row in rows)

    reference_gaps = [
        row.reference_gap for row in rows if row.reference_gap is not None
    ]
    mean_reference_gap = (
        fmean(reference_gaps) if reference_gaps else None
    )

    fitness = -mean_excess_ratio - complexity_penalty * complexity

    return Evaluation(
        expression=expression,
        cases=tuple(rows),
        mean_bins=fmean(row.bins_used for row in rows),
        mean_excess=fmean(row.excess_bins for row in rows),
        mean_excess_ratio=mean_excess_ratio,
        best_fit_wins=wins,
        best_fit_ties=ties,
        best_fit_losses=losses,
        complexity=complexity,
        fitness=fitness,
        mean_reference_gap=mean_reference_gap,
    )


def aggregate_by_family(evaluation: Evaluation) -> dict[str, dict[str, float | None]]:
    families = sorted({row.family for row in evaluation.cases})
    result: dict[str, dict[str, float | None]] = {}

    for family in families:
        rows = [row for row in evaluation.cases if row.family == family]
        reference_gaps = [
            row.reference_gap for row in rows if row.reference_gap is not None
        ]
        result[family] = {
            "mean_bins": fmean(row.bins_used for row in rows),
            "mean_excess_ratio": fmean(row.excess_ratio for row in rows),
            "mean_best_fit_bins": fmean(row.best_fit_bins for row in rows),
            "mean_reference_gap": (
                fmean(reference_gaps) if reference_gaps else None
            ),
        }

    return result
