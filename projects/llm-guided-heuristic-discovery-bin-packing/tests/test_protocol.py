from llm_binpack.protocol import standard_protocol


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
