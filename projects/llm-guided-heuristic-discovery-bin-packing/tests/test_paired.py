import pytest

from llm_binpack.evaluator import CaseResult, Evaluation
from llm_binpack.paired import compare_to_best_fit


def _evaluation_with_four_wins() -> Evaluation:
    cases = tuple(
        CaseResult(
            family="test",
            name=f"case-{i}",
            source="unit-test",
            bins_used=9,
            lower_bound=8,
            best_fit_bins=10,
        )
        for i in range(4)
    )
    return Evaluation(
        expression="test",
        cases=cases,
        mean_bins=9.0,
        mean_excess=1.0,
        mean_excess_ratio=0.125,
        best_fit_wins=4,
        best_fit_ties=0,
        best_fit_losses=0,
        complexity=1,
        fitness=-0.125,
    )


def test_exact_sign_test_and_bootstrap_direction() -> None:
    result = compare_to_best_fit(
        _evaluation_with_four_wins(),
        bootstrap_resamples=200,
        bootstrap_seed=1,
    )

    assert result.mean_delta_bins == -1.0
    assert result.wins == 4
    assert result.losses == 0
    assert result.exact_sign_pvalue == pytest.approx(0.125)
    assert result.bootstrap_mean_delta_ci95 == (-1.0, -1.0)
