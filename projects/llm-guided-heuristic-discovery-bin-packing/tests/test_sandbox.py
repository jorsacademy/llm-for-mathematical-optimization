import pytest

from llm_binpack.benchmarks import make_suite
from llm_binpack.sandbox import (
    SandboxEvaluationError,
    SandboxTimeout,
    evaluate_expression_isolated,
)


def test_isolated_evaluation_matches_expected_result() -> None:
    suite = make_suite(base_seed=44, instances_per_family=1, n_items=25)

    result = evaluate_expression_isolated(
        "-residual",
        suite,
        timeout_seconds=5.0,
    )

    assert result.evaluation.expression == "-residual"
    assert result.evaluation.best_fit_losses == 0
    assert result.wall_seconds >= 0.0


def test_isolated_evaluation_enforces_wall_clock_timeout() -> None:
    suite = make_suite(base_seed=45, instances_per_family=1, n_items=25)

    with pytest.raises(SandboxTimeout):
        evaluate_expression_isolated(
            "-residual",
            suite,
            timeout_seconds=1e-6,
        )


def test_isolated_evaluation_contains_runtime_failure() -> None:
    suite = make_suite(base_seed=46, instances_per_family=1, n_items=25)

    with pytest.raises(SandboxEvaluationError, match="ZeroDivisionError"):
        evaluate_expression_isolated(
            "1 / (residual - residual)",
            suite,
            timeout_seconds=5.0,
        )
