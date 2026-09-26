from llm_binpack.benchmarks import make_suite
from llm_binpack.evaluator import evaluate_expression
from llm_binpack.proposers import DeterministicProposer
from llm_binpack.search import SearchEngine


def test_evaluation_is_deterministic() -> None:
    suite = make_suite(base_seed=123, instances_per_family=2, n_items=40)

    first = evaluate_expression("-residual", suite)
    second = evaluate_expression("-residual", suite)

    assert first.as_dict() == second.as_dict()
    assert first.best_fit_losses == 0


def test_search_smoke() -> None:
    suite = make_suite(base_seed=321, instances_per_family=1, n_items=30)
    engine = SearchEngine(
        suite=suite,
        proposer=DeterministicProposer(seed=7),
        generations=2,
        population_size=8,
        proposals_per_generation=8,
        seed=7,
        candidate_timeout_seconds=5.0,
    )

    result = engine.run()

    assert result.best.expression
    assert len(result.history) == 3
    assert len(result.population) <= 8
    assert result.candidate_evaluations > 0
    assert result.candidate_timeouts == 0
    assert result.primary_proposer_usage["provider"] == "deterministic"
    assert result.primary_proposer_usage["calls"] == 2
