"""Evolutionary archive/search loop for heuristic discovery."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from .benchmarks import BenchmarkCase
from .evaluator import Evaluation
from .expressions import UnsafeExpression
from .proposers import DeterministicProposer, Elite, Proposer
from .sandbox import (\n    SandboxEvaluationError,\n    SandboxTimeout,\n    evaluate_expression_isolated,\n)
from .telemetry import ProposalBudgetExceeded


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

CandidateDisposition = Literal["accepted", "duplicate", "invalid"]


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
    duplicate_candidates: int
    candidate_evaluations: int
    candidate_timeouts: int
    candidate_failures: int
    candidate_evaluation_seconds: float
    proposal_failures: int
    proposal_budget_exhaustions: int
    primary_proposer_usage: dict[str, object]
    fallback_proposer_usage: dict[str, object]


def _usage_payload(proposer: object) -> dict[str, object]:
    usage_summary = getattr(proposer, "usage_summary", None)
    if not callable(usage_summary):
        return {
            "provider": type(proposer).__name__,
            "telemetry_available": False,
        }
    summary = usage_summary()
    as_dict = getattr(summary, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    return {
        "provider": type(proposer).__name__,
        "telemetry_available": False,
    }


def _is_budget_exhausted(proposer: object) -> bool:
    usage_summary = getattr(proposer, "usage_summary", None)
    if not callable(usage_summary):
        return False
    return bool(getattr(usage_summary(), "budget_exhausted", False))


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
        candidate_timeout_seconds: float = 10.0,
    ) -> None:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if candidate_timeout_seconds <= 0:
            raise ValueError("candidate_timeout_seconds must be positive")

        self.suite = suite
        self.proposer = proposer
        self.generations = generations
        self.population_size = population_size
        self.proposals_per_generation = proposals_per_generation or max(16, population_size * 2)
        self.fallback = DeterministicProposer(seed=seed + 7_919)
        self.candidate_timeout_seconds = candidate_timeout_seconds

        self._candidate_evaluations = 0
        self._candidate_timeouts = 0
        self._candidate_failures = 0
        self._candidate_evaluation_seconds = 0.0

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

    def _try_evaluate(
        self,
        expression: str,
        archive: dict[str, Evaluation],
    ) -> CandidateDisposition:
        normalized = " ".join(expression.strip().split())
        if not normalized or normalized in archive:
            return "duplicate"

        self._candidate_evaluations += 1
        started = perf_counter()
        try:
            sandboxed = evaluate_expression_isolated(
                normalized,
                self.suite,
                timeout_seconds=self.candidate_timeout_seconds,
            )
            archive[normalized] = sandboxed.evaluation
        except SandboxTimeout:
            self._candidate_timeouts += 1
            return "invalid"
        except (UnsafeExpression, SandboxEvaluationError, ValueError, OverflowError):
            self._candidate_failures += 1
            return "invalid"
        finally:
            self._candidate_evaluation_seconds += perf_counter() - started
        return "accepted"

    @staticmethod
    def _count_disposition(
        disposition: CandidateDisposition,
        *,
        invalid_candidates: int,
        duplicate_candidates: int,
    ) -> tuple[int, int]:
        if disposition == "invalid":
            invalid_candidates += 1
        elif disposition == "duplicate":
            duplicate_candidates += 1
        return invalid_candidates, duplicate_candidates

    def run(self) -> SearchResult:
        archive: dict[str, Evaluation] = {}
        invalid_candidates = 0
        duplicate_candidates = 0
        proposal_failures = 0
        proposal_budget_exhaustions = 0
        primary_disabled = False

        for expression in SEED_EXPRESSIONS:
            disposition = self._try_evaluate(expression, archive)
            invalid_candidates, duplicate_candidates = self._count_disposition(
                disposition,
                invalid_candidates=invalid_candidates,
                duplicate_candidates=duplicate_candidates,
            )

        history: list[GenerationRecord] = []

        for generation in range(self.generations + 1):
            ranked = sorted(archive.values(), key=lambda e: e.fitness, reverse=True)
            population = ranked[: self.population_size]
            if not population:
                raise RuntimeError("no valid heuristic candidates were available")
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

            proposals: list[str] = []
            if not primary_disabled:
                try:
                    proposals = self.proposer.propose(
                        elites,
                        self.proposals_per_generation,
                    )
                    if _is_budget_exhausted(self.proposer):
                        primary_disabled = True
                        proposal_budget_exhaustions += 1
                except ProposalBudgetExceeded:
                    primary_disabled = True
                    proposal_budget_exhaustions += 1
                except Exception:
                    proposal_failures += 1

            if len(proposals) < self.proposals_per_generation:
                proposals.extend(
                    self.fallback.propose(
                        elites,
                        self.proposals_per_generation - len(proposals),
                    )
                )

            accepted = 0
            for expression in proposals:
                disposition = self._try_evaluate(expression, archive)
                if disposition == "accepted":
                    accepted += 1
                invalid_candidates, duplicate_candidates = self._count_disposition(
                    disposition,
                    invalid_candidates=invalid_candidates,
                    duplicate_candidates=duplicate_candidates,
                )

            if accepted == 0:
                for expression in self.fallback.propose(elites, self.population_size):
                    disposition = self._try_evaluate(expression, archive)
                    invalid_candidates, duplicate_candidates = self._count_disposition(
                        disposition,
                        invalid_candidates=invalid_candidates,
                        duplicate_candidates=duplicate_candidates,
                    )

        ranked = sorted(archive.values(), key=lambda e: e.fitness, reverse=True)
        population = tuple(ranked[: self.population_size])
        return SearchResult(
            best=population[0],
            population=population,
            history=tuple(history),
            invalid_candidates=invalid_candidates,
            duplicate_candidates=duplicate_candidates,
            candidate_evaluations=self._candidate_evaluations,
            candidate_timeouts=self._candidate_timeouts,
            candidate_failures=self._candidate_failures,
            candidate_evaluation_seconds=self._candidate_evaluation_seconds,
            proposal_failures=proposal_failures,
            proposal_budget_exhaustions=proposal_budget_exhaustions,
            primary_proposer_usage=_usage_payload(self.proposer),
            fallback_proposer_usage=_usage_payload(self.fallback),
        )
