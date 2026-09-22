import pytest

from llm_binpack.expressions import UnsafeExpression, compile_expression


CONTEXT = {
    "item": 0.4,
    "capacity": 1.0,
    "remaining": 0.7,
    "residual": 0.3,
    "fill_ratio": 0.3,
    "item_ratio": 0.4,
    "tightness": 0.7,
    "step_fraction": 0.2,
    "open_bins": 3.0,
}


def test_valid_expression_compiles_and_runs() -> None:
    heuristic = compile_expression("-residual + 0.1 * fill_ratio")
    assert heuristic(CONTEXT) == pytest.approx(-0.27)


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('echo unsafe')",
        "(1).__class__",
        "open_bins[0]",
        "(lambda x: x)(item)",
        "residual ** 99",
    ],
)
def test_unsafe_constructs_are_rejected(source: str) -> None:
    with pytest.raises(UnsafeExpression):
        compile_expression(source)
