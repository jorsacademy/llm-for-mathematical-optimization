from llm_binpack.benchmarks import make_suite
from llm_binpack.evaluator import evaluate_expression
from llm_binpack.protocol import select_population_by_validation, standard_protocol


def test_protocol_has_disjoint_seeds_and_scale_shift() -> None:
    protocol = standard_protocol(
        instances_per_family=2,
        train_items=20,
        validation_items=20,
        generalization_items=50,
    )

    train_seeds = {case.seed for case in protocol.train}
    validation_seeds = {case.seed for case in protocol.validation}
    generalization_seeds = {case.seed for case in protocol.generalization}

    assert train_seeds.isdisjoint(validation_seeds)
    assert train_seeds.isdisjoint(generalization_seeds)
    assert validation_seeds.isdisjoint(generalization_seeds)

    assert {len(case.instance.items) for case in protocol.train} == {20}
    assert {len(case.instance.items) for case in protocol.validation} == {20}
    assert {len(case.instance.items) for case in protocol.generalization} == {50}


def test_selection_uses_validation_fitness_only() -> None:
    train = make_suite(base_seed=101, instances_per_family=1, n_items=20)
    validation = make_suite(base_seed=202, instances_per_family=1, n_items=20)
    population = (
        evaluate_expression("-residual", train),
        evaluate_expression("remaining", train),
    )

    selection = select_population_by_validation(
        population,
        validation,
        timeout_seconds=5.0,
    )
    direct_scores = {
        candidate.expression: evaluate_expression(candidate.expression, validation).fitness
        for candidate in population
    }

    assert selection.validation_evaluation.fitness == max(direct_scores.values())
    assert selection.training_evaluation.expression == selection.validation_evaluation.expression
