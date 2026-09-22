"""Command-line interface for benchmark evaluation and discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import holdout_suite, training_suite
from .evaluator import aggregate_by_family, evaluate_expression
from .orlib import load_orlib
from .paired import compare_to_best_fit
from .proposers import DeterministicProposer, OpenAIProposer
from .protocol import standard_protocol
from .search import SearchEngine


def _write_json(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluation_payload(expression: str, suite) -> dict[str, object]:
    evaluation = evaluate_expression(expression, suite)
    return {
        "evaluation": evaluation.as_dict(),
        "paired_vs_best_fit": compare_to_best_fit(evaluation).as_dict(),
        "by_family": aggregate_by_family(evaluation),
    }


def _benchmark(args: argparse.Namespace) -> int:
    suite = holdout_suite(
        seed=args.seed,
        instances_per_family=args.instances,
        n_items=args.items,
    )
    payload = _evaluation_payload(args.expression, suite)
    print(json.dumps(payload, indent=2))
    _write_json(args.output, payload)
    return 0


def _compare(args: argparse.Namespace) -> int:
    if args.orlib:
        suite = load_orlib(args.orlib)
        suite_name = f"or-library:{Path(args.orlib).name}"
    else:
        protocol = standard_protocol(instances_per_family=args.instances)
        suite = getattr(protocol, args.suite)
        suite_name = args.suite

    evaluation = evaluate_expression(args.expression, suite)
    paired = compare_to_best_fit(
        evaluation,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = {
        "suite": suite_name,
        "evaluation": evaluation.as_dict(),
        "paired_vs_best_fit": paired.as_dict(),
        "by_family": aggregate_by_family(evaluation),
    }
    print(json.dumps(payload, indent=2))
    _write_json(args.output, payload)
    return 0


def _protocol(args: argparse.Namespace) -> int:
    protocol = standard_protocol(
        instances_per_family=args.instances,
        train_items=args.train_items,
        validation_items=args.validation_items,
        generalization_items=args.generalization_items,
    )

    payload: dict[str, object] = {
        "expression": args.expression,
        "policy": {
            "train": "search only",
            "validation": "selection/diagnostics",
            "generalization": "final frozen-candidate reporting",
        },
    }

    for split_name in ("train", "validation", "generalization"):
        suite = getattr(protocol, split_name)
        evaluation = evaluate_expression(args.expression, suite)
        payload[split_name] = {
            "evaluation": evaluation.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(
                evaluation,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            ).as_dict(),
            "by_family": aggregate_by_family(evaluation),
        }

    if args.orlib:
        suite = load_orlib(args.orlib)
        evaluation = evaluate_expression(args.expression, suite)
        payload["external_or_library"] = {
            "file": str(args.orlib),
            "evaluation": evaluation.as_dict(),
            "paired_vs_best_fit": compare_to_best_fit(
                evaluation,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            ).as_dict(),
            "by_family": aggregate_by_family(evaluation),
        }

    print(json.dumps(payload, indent=2))
    _write_json(args.output, payload)
    return 0


def _discover(args: argparse.Namespace) -> int:
    train = training_suite(
        seed=args.seed,
        instances_per_family=args.instances,
        n_items=args.items,
    )

    if args.provider == "openai":
        if not args.model:
            raise SystemExit("--model is required when --provider=openai")
        proposer = OpenAIProposer(model=args.model)
    else:
        proposer = DeterministicProposer(seed=args.seed)

    engine = SearchEngine(
        suite=train,
        proposer=proposer,
        generations=args.generations,
        population_size=args.population,
        proposals_per_generation=args.proposals,
        seed=args.seed,
    )
    search = engine.run()

    holdout = holdout_suite(
        seed=args.holdout_seed,
        instances_per_family=args.holdout_instances,
        n_items=args.holdout_items,
    )
    holdout_eval = evaluate_expression(search.best.expression, holdout)

    payload = {
        "provider": args.provider,
        "model": args.model,
        "training_best": search.best.as_dict(),
        "training_paired_vs_best_fit": compare_to_best_fit(search.best).as_dict(),
        "training_by_family": aggregate_by_family(search.best),
        "holdout": holdout_eval.as_dict(),
        "holdout_paired_vs_best_fit": compare_to_best_fit(holdout_eval).as_dict(),
        "holdout_by_family": aggregate_by_family(holdout_eval),
        "invalid_candidates": search.invalid_candidates,
        "history": [
            {
                "generation": row.generation,
                "archive_size": row.archive_size,
                "best_expression": row.best_expression,
                "best_fitness": row.best_fitness,
                "best_mean_excess_ratio": row.best_mean_excess_ratio,
            }
            for row in search.history
        ],
    }

    print(json.dumps(payload, indent=2))
    _write_json(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-binpack",
        description="Discover interpretable online bin-packing heuristics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="evaluate one expression")
    benchmark.add_argument("--expression", default="-residual")
    benchmark.add_argument("--seed", type=int, default=10_017)
    benchmark.add_argument("--instances", type=int, default=12)
    benchmark.add_argument("--items", type=int, default=500)
    benchmark.add_argument("--output")
    benchmark.set_defaults(func=_benchmark)

    compare = subparsers.add_parser(
        "compare",
        help="paired comparison against Best Fit on one benchmark suite",
    )
    compare.add_argument("--expression", required=True)
    compare.add_argument(
        "--suite",
        choices=("train", "validation", "generalization"),
        default="validation",
    )
    compare.add_argument("--orlib", help="path to OR-Library binpack*.txt")
    compare.add_argument("--instances", type=int, default=12)
    compare.add_argument("--bootstrap-resamples", type=int, default=2_000)
    compare.add_argument("--bootstrap-seed", type=int, default=73)
    compare.add_argument("--output")
    compare.set_defaults(func=_compare)

    protocol = subparsers.add_parser(
        "protocol",
        help="evaluate a frozen expression on train/validation/generalization",
    )
    protocol.add_argument("--expression", required=True)
    protocol.add_argument("--instances", type=int, default=12)
    protocol.add_argument("--train-items", type=int, default=200)
    protocol.add_argument("--validation-items", type=int, default=200)
    protocol.add_argument("--generalization-items", type=int, default=500)
    protocol.add_argument("--orlib", help="optional external OR-Library file")
    protocol.add_argument("--bootstrap-resamples", type=int, default=2_000)
    protocol.add_argument("--bootstrap-seed", type=int, default=73)
    protocol.add_argument("--output", default="results/protocol.json")
    protocol.set_defaults(func=_protocol)

    discover = subparsers.add_parser("discover", help="run heuristic program search")
    discover.add_argument(
        "--provider",
        choices=("deterministic", "openai"),
        default="deterministic",
    )
    discover.add_argument("--model")
    discover.add_argument("--generations", type=int, default=20)
    discover.add_argument("--population", type=int, default=24)
    discover.add_argument("--proposals", type=int, default=None)
    discover.add_argument("--seed", type=int, default=17)
    discover.add_argument("--instances", type=int, default=12)
    discover.add_argument("--items", type=int, default=200)
    discover.add_argument("--holdout-seed", type=int, default=10_017)
    discover.add_argument("--holdout-instances", type=int, default=12)
    discover.add_argument("--holdout-items", type=int, default=500)
    discover.add_argument("--output", default="results/latest.json")
    discover.set_defaults(func=_discover)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
