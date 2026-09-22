"""Frozen scale-generalization evaluation on OR-Library binpack2-4.

No search, validation, or heuristic tuning occurs in this script.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from llm_binpack.evaluator import evaluate_expression
from llm_binpack.orlib import parse_orlib_text
from llm_binpack.paired import compare_to_best_fit


FROZEN_EXPRESSION = (
    "(2.0 + tightness) if residual < 0.25 * item else remaining / capacity"
)

ORLIB_FILES = {
    "binpack2.txt": {
        "items_per_instance": 250,
        "url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack2.txt",
        "mirror": (
            "https://raw.githubusercontent.com/ultrabababa/FunSearch-DEC/"
            "a7b3257b0b7b8f54ab6cc1fc64cadbf00e486032/dataset/binpack2.txt"
        ),
    },
    "binpack3.txt": {
        "items_per_instance": 500,
        "url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack3.txt",
        "mirror": (
            "https://raw.githubusercontent.com/ultrabababa/FunSearch-DEC/"
            "a7b3257b0b7b8f54ab6cc1fc64cadbf00e486032/dataset/binpack3.txt"
        ),
    },
    "binpack4.txt": {
        "items_per_instance": 1000,
        "url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack4.txt",
        "mirror": (
            "https://raw.githubusercontent.com/ultrabababa/FunSearch-DEC/"
            "a7b3257b0b7b8f54ab6cc1fc64cadbf00e486032/dataset/binpack4.txt"
        ),
    },
}


def download_text(filename: str) -> tuple[str, str]:
    meta = ORLIB_FILES[filename]
    errors: list[str] = []
    for url in (meta["url"], meta["mirror"]):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read().decode("utf-8"), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"could not download {filename}: " + " | ".join(errors))


def evaluate_file(filename: str) -> dict[str, object]:
    text, source_url = download_text(filename)
    suite = parse_orlib_text(text, source=f"or-library:{filename}")
    if len(suite) != 20:
        raise RuntimeError(f"expected 20 instances in {filename}, got {len(suite)}")

    evaluation = evaluate_expression(FROZEN_EXPRESSION, suite)
    paired = compare_to_best_fit(
        evaluation,
        bootstrap_resamples=10_000,
        bootstrap_seed=20260919,
    )

    return {
        "file": filename,
        "source_url": source_url,
        "instances": len(suite),
        "items_per_instance": ORLIB_FILES[filename]["items_per_instance"],
        "expression": FROZEN_EXPRESSION,
        "evaluation": evaluation.as_dict(),
        "paired_vs_best_fit": paired.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/frozen_scale_generalization.json",
    )
    args = parser.parse_args()

    payload = {
        "protocol": (
            "Frozen heuristic selected before binpack3/binpack4 evaluation. "
            "This script performs evaluation only and contains no search or tuning."
        ),
        "frozen_expression": FROZEN_EXPRESSION,
        "datasets": {
            filename: evaluate_file(filename)
            for filename in ("binpack2.txt", "binpack3.txt", "binpack4.txt")
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
