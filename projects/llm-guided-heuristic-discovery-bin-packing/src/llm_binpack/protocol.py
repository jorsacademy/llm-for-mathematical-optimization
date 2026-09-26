"""Canonical train/validation/generalization protocol."""

from __future__ import annotations

from dataclasses import dataclass

from .benchmarks import BenchmarkCase, make_suite
from .evaluator import Evaluation
from .sandbox import SandboxEvaluationError, evaluate_expression_isolated


@dataclass(frozen=True, slots=True)
class ProtocolSuites:
    train: tuple[BenchmarkCase, ...]
    validation: tuple[BenchmarkCase, ...]
    generalization: tuple[BenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    training_evaluation: Evaluation
    validation_evaluation: Evaluation
    failed_validation_evaluations: int


def standard_protocol(
    *,
    instances_per_family: int = 12,
    train_items: int = 200,
    validation_items: int = 200,
    generalization_items: int = 500,
) -> ProtocolSuites:
    """Return non-overlapping deterministic benchmark partitions.

    Search must only consume train. Heuristic/model selection may inspect
    validation. Generalization is reserved for final reporting after the
    candidate is frozen.
    """

    return ProtocolSuites(
        train=make_suite(
            base_seed=17,
            instances_per_family=instances_per_family,
            n_items=train_items,
        ),
        validation=make_suite(
            base_seed=10_017,
            instances_per_family=instances_per_family,
            n_items=validation_items,
        ),
        generalization=make_suite(
            base_seed=20_017,
            instances_per_family=instances_per_family,
            n_items=generalization_items,
        ),
    )


def select_population_by_validation(
    population: tuple[Evaluation, ...],
    validation: tuple[BenchmarkCase, ...],
    *,
    timeout_seconds: float = 10.0,
) -> ValidationSelection:
    """Select only among train-discovered candidates using validation fitness."""

    if not population:
        raise ValueError("population must not be empty")

    scored: list[tuple[Evaluation, Evaluation]] = []
    failures = 0
    for training_evaluation in population:
        try:
            validation_result = evaluate_expression_isolated(
                training_evaluation.expression,
                validation,
                timeout_seconds=timeout_seconds,
            )
        except SandboxEvaluationError:
            failures += 1
            continue
        scored.append((training_evaluation, validation_result.evaluation))

    if not scored:
        raise RuntimeError("all validation candidate evaluations failed")

    selected_training, selected_validation = max(
        scored,
        key=lambda pair: pair[1].fitness,
    )
    return ValidationSelection(
        training_evaluation=selected_training,
        validation_evaluation=selected_validation,
        failed_validation_evaluations=failures,
    )
