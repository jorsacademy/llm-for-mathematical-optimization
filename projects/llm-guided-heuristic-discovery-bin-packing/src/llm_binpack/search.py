"""Evolutionary archive/search loop for heuristic discovery."""

from __future__ import annotations

from dataclasses import dataclass

from .benchmarks import BenchmarkCase
from .evaluator import Evaluation, evaluate_expression
from .expressions import UnsafeExpression
from .proposers import DeterministicProposer, Elite, Proposer


SEED_EXPRESSIONS = (
    "-residual",
    "-(residual ** 2)",
    "-abs(residual)",
    "tightness",
    "tightness + 0.05 * fill_ratio",
    "-residual + 0.05 * item_ratio",
    "-residual - 0.05 * fill_ratio",
    "tightness - 0.02 * open_bins",
)


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    archive_size: int
    best_expression: str
    best_fitness: float
    best_mean_excess_ratio: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: Evaluation
    population: tuple[Evaluation, ...]
    history: tuple[GenerationRecord, ...]
    invalid_candidates: int


class SearchEngine:
    def __init__(
        self,
        *,
        suite: tuple[BenchmarkCase, ...],
        proposer: Proposer,
        generations: int = 20,
        population_size: int = 24,
        proposals_per_generation: int | None = None,
        seed: int = 0,
    ) -> None:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        if population_size < 2:
            raise ValueError("population_size must be at least 2")

        self.suite = suite
        self.proposer = proposer
        self.generations = generations
        self.population_size = population_size
        self.proposals_per_generation = proposals_per_generation or max(16, population_size * 2)
        self.fallback = DeterministicProposer(seed=seed + 7_919)

    @staticmethod
    def _elite_view(evaluations: list[Evaluation]) -> list[Elite]:
        return [
            Elite(
                expression=e.expression,
                fitness=e.fitness,
                mean_excess_ratio=e.mean_excess_ratio,
                wins=e.best_fit_wins,
                ties=e.best_fit_ties,
                losses=e.best_fit_losses,
            )
            for e in evaluations
        ]

    def _try_evaluate(self, expression: str, archive: dict[str, Evaluation]) -> bool:
        normalized = " ".join(expression.strip().split())
        if not normalized or normalized in archive:
            return False
        try:
            archive[normalized] = evaluate_expression(normalized, self.suite)
        except (UnsafeExpression, ValueError, ZeroDivisionError, OverflowError):
            return False
        return True

    def run(self) -> SearchResult:
        archive: dict[str, Evaluation] = {}
        invalid_candidates = 0

        for expression in SEED_EXPRESSIONS:
            if not self._try_evaluate(expression, archive):
                invalid_candidates += 1

        history: list[GenerationRecord] = []

        for generation in range(self.generations + 1):
            ranked = sorted(archive.values(), key=lambda e: e.fitness, reverse=True)
            population = ranked[: self.population_size]
            best = population[0]

            history.append(
                GenerationRecord(
                    generation=generation,
                    archive_size=len(archive),
                    best_expression=best.expression,
                    best_fitness=best.fitness,
                    best_mean_excess_ratio=best.mean_excess_ratio,
                )
            )

            if generation == self.generations:
                break

            elite_count = max(2, min(8, len(population) // 3 or 2))
            elites = self._elite_view(population[:elite_count])

            try:
                proposals = self.proposer.propose(elites, self.proposals_per_generation)
            except Exception:
                proposals = []

            if len(proposals) < self.proposals_per_generation:
                proposals.extend(
                    self.fallback.propose(
                        elites,
                        self.proposals_per_generation - len(proposals),
                    )
                )

            accepted = 0
            for expression in proposals:
                if self._try_evaluate(expression, archive):
                    accepted += 1
                else:
                    invalid_candidates += 1

            if accepted == 0:
                for expression in self.fallback.propose(elites, self.population_size):
                    if not self._try_evaluate(expression, archive):
                        invalid_candidates += 1

        ranked = sorted(archive.values(), key=lambda e: e.fitness, reverse=True)
        population = tuple(ranked[: self.population_size])
        return SearchResult(
            best=population[0],
            population=population,
            history=tuple(history),
            invalid_candidates=invalid_candidates,
        )
