"""Usage, budget, and optional cost telemetry for proposal backends."""

from __future__ import annotations

from dataclasses import dataclass


class ProposalBudgetExceeded(RuntimeError):
    """Raised before an LLM call when the configured proposal budget is exhausted."""


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """User-supplied token prices used only for transparent cost estimation."""

    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None

    def __post_init__(self) -> None:
        configured = (
            self.input_usd_per_million is not None,
            self.output_usd_per_million is not None,
        )
        if configured[0] != configured[1]:
            raise ValueError("input and output token prices must be provided together")
        for value in configured_values(self):
            if value < 0:
                raise ValueError("token prices must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.input_usd_per_million is not None

    def estimate(self, input_tokens: int, output_tokens: int) -> float | None:
        if not self.enabled:
            return None
        assert self.input_usd_per_million is not None
        assert self.output_usd_per_million is not None
        return (
            input_tokens * self.input_usd_per_million
            + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    def as_dict(self) -> dict[str, float | None]:
        return {
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
        }


def configured_values(pricing: TokenPricing) -> tuple[float, ...]:
    return tuple(
        value
        for value in (
            pricing.input_usd_per_million,
            pricing.output_usd_per_million,
        )
        if value is not None
    )


@dataclass(frozen=True, slots=True)
class ProposalBudget:
    """Hard call budget and post-call token budget for an optional real LLM."""

    max_calls: int | None = None
    max_total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_calls is not None and self.max_calls <= 0:
            raise ValueError("max_calls must be positive when provided")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive when provided")

    def as_dict(self) -> dict[str, int | None]:
        return {
            "max_calls": self.max_calls,
            "max_total_tokens": self.max_total_tokens,
        }


@dataclass(frozen=True, slots=True)
class UsageRecord:
    provider: str
    model: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    requested_proposals: int
    returned_proposals: int
    success: bool
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "requested_proposals": self.requested_proposals,
            "returned_proposals": self.returned_proposals,
            "success": self.success,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class UsageSummary:
    provider: str
    model: str | None
    calls: int
    successful_calls: int
    failed_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    requested_proposals: int
    returned_proposals: int
    latency_seconds: float
    estimated_cost_usd: float | None
    budget: ProposalBudget
    pricing: TokenPricing
    budget_exhausted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requested_proposals": self.requested_proposals,
            "returned_proposals": self.returned_proposals,
            "latency_seconds": self.latency_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
            "budget": self.budget.as_dict(),
            "pricing": self.pricing.as_dict(),
            "budget_exhausted": self.budget_exhausted,
        }


class UsageLedger:
    """Mutable ledger owned by one proposer instance."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None = None,
        budget: ProposalBudget | None = None,
        pricing: TokenPricing | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.budget = budget or ProposalBudget()
        self.pricing = pricing or TokenPricing()
        self._records: list[UsageRecord] = []

    def require_capacity(self) -> None:
        summary = self.summary()
        if self.budget.max_calls is not None and summary.calls >= self.budget.max_calls:
            raise ProposalBudgetExceeded("maximum LLM call budget reached")
        if (
            self.budget.max_total_tokens is not None
            and summary.total_tokens >= self.budget.max_total_tokens
        ):
            raise ProposalBudgetExceeded("maximum reported token budget reached")

    def record(
        self,
        *,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        latency_seconds: float,
        requested_proposals: int,
        returned_proposals: int,
        success: bool,
        error: str | None = None,
    ) -> None:
        resolved_total = (
            input_tokens + output_tokens if total_tokens is None else total_tokens
        )
        values = (
            input_tokens,
            cached_input_tokens,
            output_tokens,
            resolved_total,
            requested_proposals,
            returned_proposals,
        )
        if any(value < 0 for value in values):
            raise ValueError("usage counters must be non-negative")
        if latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")

        self._records.append(
            UsageRecord(
                provider=self.provider,
                model=self.model,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                total_tokens=resolved_total,
                latency_seconds=latency_seconds,
                requested_proposals=requested_proposals,
                returned_proposals=returned_proposals,
                success=success,
                error=error,
            )
        )

    def summary(self) -> UsageSummary:
        calls = len(self._records)
        input_tokens = sum(record.input_tokens for record in self._records)
        cached_tokens = sum(record.cached_input_tokens for record in self._records)
        output_tokens = sum(record.output_tokens for record in self._records)
        total_tokens = sum(record.total_tokens for record in self._records)
        successful = sum(record.success for record in self._records)
        estimated_cost = self.pricing.estimate(input_tokens, output_tokens)

        exhausted = False
        if self.budget.max_calls is not None and calls >= self.budget.max_calls:
            exhausted = True
        if (
            self.budget.max_total_tokens is not None
            and total_tokens >= self.budget.max_total_tokens
        ):
            exhausted = True

        return UsageSummary(
            provider=self.provider,
            model=self.model,
            calls=calls,
            successful_calls=successful,
            failed_calls=calls - successful,
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            requested_proposals=sum(
                record.requested_proposals for record in self._records
            ),
            returned_proposals=sum(
                record.returned_proposals for record in self._records
            ),
            latency_seconds=sum(record.latency_seconds for record in self._records),
            estimated_cost_usd=estimated_cost,
            budget=self.budget,
            pricing=self.pricing,
            budget_exhausted=exhausted,
        )

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)
