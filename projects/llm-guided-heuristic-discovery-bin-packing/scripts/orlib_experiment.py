"""Reproducible OR-Library experiment for deterministic vs recorded LLM search."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from llm_binpack.evaluator import aggregate_by_family, evaluate_expression
from llm_binpack.orlib import parse_orlib_text
from llm_binpack.paired import compare_to_best_fit
from llm_binpack.proposers import DeterministicProposer, Elite
from llm_binpack.search import SearchEngine


ORLIB_URLS = {
    "binpack1.txt": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack1.txt",
    "binpack2.txt": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack2.txt",
}
MIRROR_URLS = {
    "binpack1.txt": (
        "https://raw.githubusercontent.com/ultrabababa/FunSearch-DEC/"
        "a7b3257b0b7b8f54ab6cc1fc64cadbf00e486032/dataset/binpack1.txt"
    ),
    "binpack2.txt": (
        "https://raw.githubusercontent.com/ultrabababa/FunSearch-DEC/"
        "a7b3257b0b7b8f54ab6cc1fc64cadbf00e486032/dataset/binpack2.txt"
    ),
}


class RecordedProposer:
    def __init__(self, generations: list[list[str]]) -> None:
        self.generations = generations
        self.index = 0

    def propose(self, elites: list[Elite], n: int) -> list[str]:
        del elites
        if self.index >= len(self.generations):
            return []
        batch = self.generations[self.index]
        self.index += 1
        if len(batch) != n:
            raise ValueError(
                f"recorded generation has {len(batch)} proposals; expected exactly {n}"
            )
        return list(batch)


def download_text(filename: str) -> tuple[str, str]:
    errors: list[str] = []
    for url in (ORLIB_URLS[filename], MIRROR_URLS[filename]):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8"), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"could not download {filename}: " + " | ".join(errors))


def select_on_validation(search, validation):
    scored = [
        evaluate_expression(candidate.expression, validation)
        for candidate in search.population
    ]
    return max(scored, key=lambda result: result.fitness), scored


def report_method(name, search, validation, test, external):
    selected_validation, validation_pool = select_on_validation(search, validation)
    expression = selected_validation.expression
    train_eval = evaluate_expression(expression, TRAIN)
    test_eval = evaluate_expression(expression, test)
    external_eval = evaluate_expression(expression, external)

    return {
        "method": name,
        "selected_expression": expression,
        "candidate_budget": PROPOSALS * GENERATIONS,
        "archive_size": search.history[-1].archive_size,
        "train": {
            "evaluation": train_eval.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(train_eval).as_dict(),
        },
        "validation": {
            "evaluation": selected_validation.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(selected_validation).as_dict(),
        },
        "test": {
            "evaluation": test_eval.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(test_eval).as_dict(),
        },
        "external_binpack2": {
            "evaluation": external_eval.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(external_eval).as_dict(),
        },
        "top_validation_candidates": [
            evaluation.as_dict()
            for evaluation in sorted(
                validation_pool,
                key=lambda result: result.fitness,
                reverse=True,
            )[:5]
        ],
        "search_history": [
            {
                "generation": row.generation,
                "archive_size": row.archive_size,
                "best_expression": row.best_expression,
                "best_fitness": row.best_fitness,
            }
            for row in search.history
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="experiments/orlib/llm_candidates.json",
    )
    parser.add_argument(
        "--output",
        default="results/orlib_binpack1_experiment.json",
    )
    args = parser.parse_args()

    text, source_url = download_text("binpack1.txt")
    cases = parse_orlib_text(text, source="or-library:binpack1.txt")
    if len(cases) != 20:
        raise RuntimeError(f"expected 20 binpack1 instances, got {len(cases)}")

    global TRAIN, PROPOSALS, GENERATIONS
    TRAIN = cases[:10]
    validation = cases[10:15]
    test = cases[15:20]

    external_text, external_source_url = download_text("binpack2.txt")
    external = parse_orlib_text(
        external_text,
        source="or-library:binpack2.txt",
    )
    if len(external) != 20:
        raise RuntimeError(f"expected 20 binpack2 instances, got {len(external)}")

    candidate_payload = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    generations = candidate_payload["generations"]
    if not generations:
        raise ValueError("recorded LLM proposal file has no generations")

    PROPOSALS = len(generations[0])
    if any(len(batch) != PROPOSALS for batch in generations):
        raise ValueError("all recorded generations must have equal proposal counts")
    GENERATIONS = len(generations)

    deterministic = SearchEngine(
        suite=TRAIN,
        proposer=DeterministicProposer(seed=20260919),
        generations=GENERATIONS,
        population_size=32,
        proposals_per_generation=PROPOSALS,
        seed=20260919,
    ).run()

    recorded_llm = SearchEngine(
        suite=TRAIN,
        proposer=RecordedProposer(generations),
        generations=GENERATIONS,
        population_size=32,
        proposals_per_generation=PROPOSALS,
        seed=20260919,
    ).run()

    payload = {
        "dataset": {
            "name": "OR-Library binpack1",
            "source_url": source_url,
            "instances": len(cases),
            "split": {
                "train": [case.instance.name for case in TRAIN],
                "validation": [case.instance.name for case in validation],
                "test": [case.instance.name for case in test],
            },
            "external_generalization": {
                "name": "OR-Library binpack2",
                "source_url": external_source_url,
                "instances": len(external),
                "items_per_instance": 250,
            },
        },
        "selection_rule": (
            "Search sees train only. Final expression is selected by validation "
            "fitness from the final train-ranked population. Test is evaluated once."
        ),
        "recorded_llm_metadata": {
            "model": candidate_payload.get("model"),
            "notes": candidate_payload.get("notes"),
            "generations": GENERATIONS,
            "proposals_per_generation": PROPOSALS,
        },
        "deterministic": report_method(
            "deterministic",
            deterministic,
            validation,
            test,
            external,
        ),
        "recorded_llm": report_method(
            "recorded_llm",
            recorded_llm,
            validation,
            test,
            external,
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
