"""Reproducible benchmark families for online bin packing."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .problem import BinPackingInstance


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    family: str
    seed: int
    instance: BinPackingInstance
    reference_optimum: int | None = None
    source: str = "synthetic"


def _clip_item(value: float, capacity: float) -> float:
    return min(capacity * 0.95, max(capacity * 0.05, value))


def _uniform_items(rng: random.Random, n_items: int, capacity: float) -> tuple[float, ...]:
    return tuple(rng.uniform(0.05 * capacity, 0.95 * capacity) for _ in range(n_items))


def _weibull_items(rng: random.Random, n_items: int, capacity: float) -> tuple[float, ...]:
    return tuple(
        _clip_item(rng.weibullvariate(0.38 * capacity, 1.6), capacity)
        for _ in range(n_items)
    )


def _bimodal_items(rng: random.Random, n_items: int, capacity: float) -> tuple[float, ...]:
    values: list[float] = []
    for _ in range(n_items):
        if rng.random() < 0.55:
            value = rng.gauss(0.27 * capacity, 0.06 * capacity)
        else:
            value = rng.gauss(0.68 * capacity, 0.07 * capacity)
        values.append(_clip_item(value, capacity))
    return tuple(values)


_GENERATORS = {
    "uniform": _uniform_items,
    "weibull": _weibull_items,
    "bimodal": _bimodal_items,
}


def make_suite(
    *,
    base_seed: int,
    instances_per_family: int = 12,
    n_items: int = 200,
    capacity: float = 1.0,
    families: tuple[str, ...] = ("uniform", "weibull", "bimodal"),
) -> tuple[BenchmarkCase, ...]:
    """Create a deterministic benchmark suite with independent per-case seeds."""

    if instances_per_family <= 0 or n_items <= 0:
        raise ValueError("instances_per_family and n_items must be positive")

    cases: list[BenchmarkCase] = []
    for family_index, family in enumerate(families):
        if family not in _GENERATORS:
            raise ValueError(f"unknown benchmark family: {family!r}")
        generator = _GENERATORS[family]

        for index in range(instances_per_family):
            seed = base_seed + family_index * 100_000 + index
            rng = random.Random(seed)
            items = generator(rng, n_items, capacity)
            instance = BinPackingInstance(
                items=items,
                capacity=capacity,
                name=f"{family}-{index:03d}",
            )
            cases.append(
                BenchmarkCase(
                    family=family,
                    seed=seed,
                    instance=instance,
                    source="synthetic",
                )
            )

    return tuple(cases)


def training_suite(
    *,
    seed: int = 17,
    instances_per_family: int = 12,
    n_items: int = 200,
) -> tuple[BenchmarkCase, ...]:
    return make_suite(
        base_seed=seed,
        instances_per_family=instances_per_family,
        n_items=n_items,
    )


def holdout_suite(
    *,
    seed: int = 10_017,
    instances_per_family: int = 12,
    n_items: int = 500,
) -> tuple[BenchmarkCase, ...]:
    return make_suite(
        base_seed=seed,
        instances_per_family=instances_per_family,
        n_items=n_items,
    )
