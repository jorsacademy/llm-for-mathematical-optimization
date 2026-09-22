import pytest

from llm_binpack.orlib import ORLibraryFormatError, parse_orlib_text


SAMPLE = """2
u4_00
10 4 2
6
4
6
4
u3_00
10 3 2
5
5
5
"""


def test_parse_orlib_text() -> None:
    cases = parse_orlib_text(SAMPLE, source="unit-test")

    assert len(cases) == 2
    assert cases[0].instance.name == "u4_00"
    assert cases[0].instance.capacity == 10
    assert cases[0].instance.items == (6.0, 4.0, 6.0, 4.0)
    assert cases[0].reference_optimum == 2
    assert cases[0].source == "unit-test"


def test_orlib_rejects_truncated_instance() -> None:
    with pytest.raises(ORLibraryFormatError):
        parse_orlib_text("""1
broken
10 3 2
5
5
""")
