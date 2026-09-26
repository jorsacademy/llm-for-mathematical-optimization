"""Deterministic online one-dimensional bin-packing simulator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite


ScoreFunction = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True, slots=True)
class BinPackingInstance:
    """A sequence of online items and a fixed bin capacity."""

    items: tuple[float, ...]
    capacity: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity must be a positive finite number")
        if not self.items:
            raise ValueError("items must not be empty")
        for item in self.items:
            if not isfinite(item) or item <= 0 or item > self.capacity:
                raise ValueError(
                    f"item sizes must be finite and in (0, capacity]; got {item!r}"
                )


@dataclass(frozen=True, slots=True)
class PackingResult:
    """Result of a deterministic online packing run."""

    bins_used: int
    remaining: tuple[float, ...]
    assignments: tuple[int, ...]


def _context(
    *,
    item: float,
    remaining: float,
    capacity: float,
    step: int,
    item_count: int,
    open_bins: int,
) -> dict[str, float]:
    residual = remaining - item
    fill_ratio = (capacity - remaining) / capacity
    return {
        "item": item,
        "capacity": capacity,
        "remaining": remaining,
        "residual": residual,
        "fill_ratio": fill_ratio,
        "item_ratio": item / capacity,
        "tightness": 1.0 - (residual / capacity),
        "step_fraction": step / max(1, item_count - 1),
        "open_bins": float(open_bins),
    }


def pack_with_score(instance: BinPackingInstance, score_fn: ScoreFunction) -> PackingResult:
    """Pack items online using a priority score over feasible open bins.

    The feasible bin with maximum score is selected. Ties are resolved by the
    smallest bin index, which makes evaluation deterministic.
    """

    remaining: list[float] = []
    assignments: list[int] = []

    for step, item in enumerate(instance.items):
        best_index: int | None = None
        best_score = float("-inf")

        for index, free in enumerate(remaining):
            if free + 1e-12 < item:
                continue

            context = _context(
                item=item,
                remaining=free,
                capacity=instance.capacity,
                step=step,
                item_count=len(instance.items),
                open_bins=len(remaining),
            )
            score = float(score_fn(context))
            if not isfinite(score):
                raise ValueError(f"heuristic returned a non-finite score: {score!r}")

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            remaining.append(instance.capacity - item)
            assignments.append(len(remaining) - 1)
        else:
            remaining[best_index] -= item
            if abs(remaining[best_index]) < 1e-12:
                remaining[best_index] = 0.0
            assignments.append(best_index)

    return PackingResult(
        bins_used=len(remaining),
        remaining=tuple(remaining),
        assignments=tuple(assignments),
    )
