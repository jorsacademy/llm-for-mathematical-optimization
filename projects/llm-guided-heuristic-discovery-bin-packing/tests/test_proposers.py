from types import SimpleNamespace

import pytest

from llm_binpack.proposers import OpenAIProposer
from llm_binpack.telemetry import (
    ProposalBudget,
    ProposalBudgetExceeded,
    TokenPricing,
)


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["store"] is False
        return SimpleNamespace(
            output_text='["-residual", "tightness"]',
            usage=SimpleNamespace(
                input_tokens=100,
                input_tokens_details=SimpleNamespace(cached_tokens=25),
                output_tokens=20,
                total_tokens=120,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_openai_proposer_records_usage_and_cost_without_real_api() -> None:
    client = FakeClient()
    proposer = OpenAIProposer(
        "fake-model",
        client=client,
        budget=ProposalBudget(max_calls=1, max_total_tokens=1_000),
        pricing=TokenPricing(
            input_usd_per_million=2.0,
            output_usd_per_million=4.0,
        ),
        max_output_tokens=200,
    )

    candidates = proposer.propose([], 2)
    summary = proposer.usage_summary()

    assert candidates == ["-residual", "tightness"]
    assert summary.calls == 1
    assert summary.input_tokens == 100
    assert summary.cached_input_tokens == 25
    assert summary.output_tokens == 20
    assert summary.total_tokens == 120
    assert summary.estimated_cost_usd == pytest.approx(0.00028)
    assert summary.budget_exhausted
    assert client.responses.calls == 1

    with pytest.raises(ProposalBudgetExceeded):
        proposer.propose([], 1)

    assert client.responses.calls == 1


def test_token_pricing_requires_both_rates() -> None:
    with pytest.raises(ValueError):
        TokenPricing(input_usd_per_million=1.0)
