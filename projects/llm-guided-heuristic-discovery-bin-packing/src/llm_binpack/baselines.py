"""Classical online bin-packing baselines."""

from __future__ import annotations

from .problem import BinPackingInstance, PackingResult, pack_with_score


def first_fit(instance: BinPackingInstance) -> PackingResult:
    """Place each item in the first feasible open bin."""

    remaining: list[float] = []
    assignments: list[int] = []

    for item in instance.items:
        chosen: int | None = None
        for index, free in enumerate(remaining):
            if free + 1e-12 >= item:
                chosen = index
                break

        if chosen is None:
            remaining.append(instance.capacity - item)
            assignments.append(len(remaining) - 1)
        else:
            remaining[chosen] -= item
            if abs(remaining[chosen]) < 1e-12:
                remaining[chosen] = 0.0
            assignments.append(chosen)

    return PackingResult(len(remaining), tuple(remaining), tuple(assignments))


def best_fit(instance: BinPackingInstance) -> PackingResult:
    """Choose the feasible bin leaving the smallest residual capacity."""

    return pack_with_score(instance, lambda c: -c["residual"])


def worst_fit(instance: BinPackingInstance) -> PackingResult:
    """Choose the feasible bin leaving the largest residual capacity."""

    return pack_with_score(instance, lambda c: c["residual"])
