"""Parser for the OR-Library one-dimensional bin-packing files."""

from __future__ import annotations

from pathlib import Path

from .benchmarks import BenchmarkCase
from .problem import BinPackingInstance


class ORLibraryFormatError(ValueError):
    """Raised when an OR-Library bin-packing file is malformed."""


def parse_orlib_text(text: str, *, source: str = "or-library") -> tuple[BenchmarkCase, ...]:
    """Parse the classic OR-Library binpack1..binpack8 text format.

    Format:
        number_of_instances
        instance_name
        capacity item_count best_known
        item_1
        ...
        item_n

    The third header value is stored as an offline reference optimum/best-known
    value. It is not treated as an online optimum for the fixed arrival order.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ORLibraryFormatError("empty OR-Library file")

    try:
        expected_instances = int(lines[0])
    except ValueError as exc:
        raise ORLibraryFormatError("first line must be the number of instances") from exc

    if expected_instances <= 0:
        raise ORLibraryFormatError("instance count must be positive")

    cursor = 1
    cases: list[BenchmarkCase] = []

    for index in range(expected_instances):
        if cursor >= len(lines):
            raise ORLibraryFormatError(
                f"unexpected end of file before instance {index + 1}"
            )

        name = lines[cursor]
        cursor += 1

        if cursor >= len(lines):
            raise ORLibraryFormatError(f"missing header for instance {name!r}")

        header = lines[cursor].split()
        cursor += 1
        if len(header) != 3:
            raise ORLibraryFormatError(
                f"instance {name!r} header must contain capacity, item count, best known"
            )

        try:
            capacity = float(header[0])
            item_count = int(header[1])
            reference_optimum = int(header[2])
        except ValueError as exc:
            raise ORLibraryFormatError(
                f"instance {name!r} has a non-numeric header"
            ) from exc

        if item_count <= 0:
            raise ORLibraryFormatError(f"instance {name!r} has no items")

        if cursor + item_count > len(lines):
            raise ORLibraryFormatError(
                f"instance {name!r} declares {item_count} items but file ends early"
            )

        try:
            items = tuple(float(lines[cursor + offset]) for offset in range(item_count))
        except ValueError as exc:
            raise ORLibraryFormatError(
                f"instance {name!r} contains a non-numeric item"
            ) from exc
        cursor += item_count

        instance = BinPackingInstance(
            items=items,
            capacity=capacity,
            name=name,
        )
        cases.append(
            BenchmarkCase(
                family="or-library",
                seed=index,
                instance=instance,
                reference_optimum=reference_optimum,
                source=source,
            )
        )

    if cursor != len(lines):
        raise ORLibraryFormatError(
            f"file contains {len(lines) - cursor} unexpected trailing non-empty lines"
        )

    return tuple(cases)


def load_orlib(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load an OR-Library one-dimensional bin-packing file from disk."""

    target = Path(path)
    return parse_orlib_text(
        target.read_text(encoding="utf-8"),
        source=f"or-library:{target.name}",
    )
