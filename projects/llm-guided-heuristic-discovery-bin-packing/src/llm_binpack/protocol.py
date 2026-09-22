"""Canonical train/validation/generalization protocol."""

from __future__ import annotations

from dataclasses import dataclass

from .benchmarks import BenchmarkCase, make_suite


@dataclass(frozen=True, slots=True)
class ProtocolSuites:
    train: tuple[BenchmarkCase, ...]
    validation: tuple[BenchmarkCase, ...]
    generalization: tuple[BenchmarkCase, ...]


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
