"""Command-line interface for benchmark evaluation and discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from .baselines import best_fit, first_fit, worst_fit
from .benchmarks import holdout_suite, make_suite, training_suite
from .evaluator import aggregate_by_family
from .orlib import load_orlib
from .paired import compare_to_best_fit
from .proposers import DeterministicProposer, OpenAIProposer
from .protocol import select_population_by_validation, standard_protocol
from .sandbox import evaluate_expression_isolated
from .search import SearchEngine
from .telemetry import ProposalBudget, TokenPricing


def _write_json(path: str | None, payload: dict[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _isolated_evaluation(expression: str, suite, timeout_seconds: float):
    return evaluate_expression_isolated(
        expression,
        suite,
        timeout_seconds=timeout_seconds,
    ).evaluation


def _evaluation_payload(
    expression: str,
    suite,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    evaluation = _isolated_evaluation(expression, suite, timeout_seconds)
    return {
        "evaluation": evaluation.as_dict(),
        "paired_vs_best_fit": compare_to_best_fit(evaluation).as_dict(),
        "by_family": aggregate_by_family(evaluation),
    }


def _classical_baselines(suite) -> dict[str, dict[str, float]]:
    solvers = {
        "first_fit": first_fit,
        "best_fit": best_fit,
        "worst_fit": worst_fit,
    }
    return {
        name: {
            "mean_bins": fmean(solver(case.instance).bins_used for case in suite),
        }
        for name, solver in solvers.items()
    }


def _assert_disjoint_seed_sets(*suites) -> None:
    seed_sets = [{case.seed for case in suite} for suite in suites]
    for left in range(len(seed_sets)):
        for right in range(left + 1, len(seed_sets)):
            if not seed_sets[left].isdisjoint(seed_sets[right]):
                raise ValueError("train, validation, and final holdout seeds must be disjoint")


def _benchmark(args: argparse.Namespace) -> int:
    suite = holdout_suite(
        seed=args.seed,
        instances_per_family=args.instances,
        n_items=args.items,
    )
    payload = _evaluation_payload(
        args.expression,
        suite,
        timeout_seconds=args.candidate_timeout,
    )
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

    evaluation = _isolated_evaluation(
        args.expression,
        suite,
        args.candidate_timeout,
    )
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
        evaluation = _isolated_evaluation(
            args.expression,
            suite,
            args.candidate_timeout,
        )
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
        evaluation = _isolated_evaluation(
            args.expression,
            suite,
            args.candidate_timeout,
        )
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
    validation = make_suite(
        base_seed=args.validation_seed,
        instances_per_family=args.validation_instances or args.instances,
        n_items=args.validation_items,
    )
    final_holdout = holdout_suite(
        seed=args.holdout_seed,
        instances_per_family=args.holdout_instances,
        n_items=args.holdout_items,
    )
    _assert_disjoint_seed_sets(train, validation, final_holdout)

    if args.provider == "openai":
        if not args.model:
            raise SystemExit("--model is required when --provider=openai")
        proposer = OpenAIProposer(
            model=args.model,
            budget=ProposalBudget(
                max_calls=args.max_llm_calls,
                max_total_tokens=args.max_total_tokens,
            ),
            pricing=TokenPricing(
                input_usd_per_million=args.input_cost_per_million,
                output_usd_per_million=args.output_cost_per_million,
            ),
            max_output_tokens=args.max_output_tokens,
        )
    else:
        proposer = DeterministicProposer(seed=args.seed)

    engine = SearchEngine(
        suite=train,
        proposer=proposer,
        generations=args.generations,
        population_size=args.population,
        proposals_per_generation=args.proposals,
        seed=args.seed,
        candidate_timeout_seconds=args.candidate_timeout,
    )
    search = engine.run()

    selection = select_population_by_validation(
        search.population,
        validation,
        timeout_seconds=args.candidate_timeout,
    )
    selected_expression = selection.validation_evaluation.expression

    final_evaluation = _isolated_evaluation(
        selected_expression,
        final_holdout,
        args.candidate_timeout,
    )

    random_baseline_proposer = DeterministicProposer(seed=args.seed + 1_000_003)
    random_expression = random_baseline_proposer.propose([], 1)[0]
    random_final = _isolated_evaluation(
        random_expression,
        final_holdout,
        args.candidate_timeout,
    )
    handwritten_expression = "tightness + 0.05 * fill_ratio"
    handwritten_final = _isolated_evaluation(
        handwritten_expression,
        final_holdout,
        args.candidate_timeout,
    )

    payload = {
        "provider": args.provider,
        "model": args.model,
        "selection_policy": {
            "search": "train only",
            "selection": "validation fitness among final train-ranked population",
            "final_holdout": "evaluated only after expression is frozen",
        },
        "selected_expression": selected_expression,
        "training_search_best": search.best.as_dict(),
        "selected_training": selection.training_evaluation.as_dict(),
        "validation_selected": selection.validation_evaluation.as_dict(),
        "failed_validation_evaluations": selection.failed_validation_evaluations,
        "final_holdout": final_evaluation.as_dict(),
        "final_holdout_paired_vs_best_fit": compare_to_best_fit(
            final_evaluation
        ).as_dict(),
        "final_holdout_by_family": aggregate_by_family(final_evaluation),
        "baselines": {
            "classical": _classical_baselines(final_holdout),
            "handwritten_expression": {
                "expression": handwritten_expression,
                "evaluation": handwritten_final.as_dict(),
            },
            "random_generated_expression": {
                "expression": random_expression,
                "evaluation": random_final.as_dict(),
            },
        },
        "search_budget": {
            "invalid_candidates": search.invalid_candidates,
            "duplicate_candidates": search.duplicate_candidates,
            "candidate_evaluations": search.candidate_evaluations,
            "candidate_timeouts": search.candidate_timeouts,
            "candidate_failures": search.candidate_failures,
            "candidate_evaluation_seconds": search.candidate_evaluation_seconds,
            "proposal_failures": search.proposal_failures,
            "proposal_budget_exhaustions": search.proposal_budget_exhaustions,
            "primary_proposer_usage": search.primary_proposer_usage,
            "fallback_proposer_usage": search.fallback_proposer_usage,
        },
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


def _add_timeout_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-timeout",
        type=float,
        default=10.0,
        help="wall-clock seconds allowed for one isolated heuristic evaluation",
    )


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
    _add_timeout_argument(benchmark)
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
    _add_timeout_argument(compare)
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
    _add_timeout_argument(protocol)
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
    discover.add_argument("--validation-seed", type=int, default=5_017)
    discover.add_argument("--validation-instances", type=int, default=None)
    discover.add_argument("--validation-items", type=int, default=200)
    discover.add_argument("--holdout-seed", type=int, default=10_017)
    discover.add_argument("--holdout-instances", type=int, default=12)
    discover.add_argument("--holdout-items", type=int, default=500)
    discover.add_argument("--max-llm-calls", type=int)
    discover.add_argument("--max-total-tokens", type=int)
    discover.add_argument("--max-output-tokens", type=int)
    discover.add_argument("--input-cost-per-million", type=float)
    discover.add_argument("--output-cost-per-million", type=float)
    discover.add_argument("--output", default="results/latest.json")
    _add_timeout_argument(discover)
    discover.set_defaults(func=_discover)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
