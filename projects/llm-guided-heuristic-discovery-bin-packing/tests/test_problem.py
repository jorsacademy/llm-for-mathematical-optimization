from llm_binpack.baselines import best_fit, first_fit, worst_fit
from llm_binpack.problem import BinPackingInstance


def test_classical_baselines_pack_valid_instance() -> None:
    instance = BinPackingInstance(items=(0.6, 0.4, 0.6, 0.4))

    assert first_fit(instance).bins_used == 2
    assert best_fit(instance).bins_used == 2
    assert worst_fit(instance).bins_used == 2


def test_instance_rejects_oversized_item() -> None:
    try:
        BinPackingInstance(items=(1.1,), capacity=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized item should be rejected")
