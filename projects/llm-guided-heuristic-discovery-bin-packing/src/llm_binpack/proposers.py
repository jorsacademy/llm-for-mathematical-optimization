"""Candidate-expression proposers for heuristic program search."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from .telemetry import ProposalBudget, TokenPricing, UsageLedger, UsageSummary


@dataclass(frozen=True, slots=True)
class Elite:
    expression: str
    fitness: float
    mean_excess_ratio: float
    wins: int
    ties: int
    losses: int


class Proposer(Protocol):
    def propose(self, elites: list[Elite], n: int) -> list[str]:
        """Return up to n candidate expressions."""


class DeterministicProposer:
    """Reproducible grammar-based proposer used for testing and fallback."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.usage = UsageLedger(provider="deterministic")

    def _fresh_expression(self) -> str:
        # Threshold rules mimic a useful online-packing pattern: aggressively
        # close a bin when the residual gap is small, otherwise preserve room.
        if self.rng.random() < 0.45:
            threshold = self.rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            tight = self.rng.choice(
                ["2.0 - residual", "1.0 + tightness", "2.0 + fill_ratio"]
            )
            loose = self.rng.choice(
                [
                    "remaining",
                    "residual",
                    "fill_ratio",
                    "-residual + 0.10 * fill_ratio",
                ]
            )
            return f"({tight}) if residual < ({threshold}) * capacity else ({loose})"

        primary = self.rng.choice(
            [
                "-residual",
                "tightness",
                "-(residual ** 2)",
                "-abs(residual)",
                "fill_ratio",
                "item_ratio",
            ]
        )
        secondary = self.rng.choice(
            [
                "fill_ratio",
                "item_ratio",
                "tightness",
                "step_fraction",
                "open_bins",
                "residual",
            ]
        )
        coefficient = self.rng.choice(
            [-0.50, -0.25, -0.10, -0.05, 0.05, 0.10, 0.25, 0.50]
        )
        return f"({primary}) + ({coefficient}) * ({secondary})"

    def _mutate_elite(self, expression: str) -> str:
        term = self.rng.choice(
            [
                "fill_ratio",
                "item_ratio",
                "tightness",
                "step_fraction",
                "residual",
                "open_bins",
            ]
        )
        coefficient = self.rng.choice([-0.20, -0.10, -0.05, 0.05, 0.10, 0.20])
        if self.rng.random() < 0.35:
            return f"({expression}) + ({coefficient}) * ({term} ** 2)"
        return f"({expression}) + ({coefficient}) * ({term})"

    def propose(self, elites: list[Elite], n: int) -> list[str]:
        started = perf_counter()
        candidates: list[str] = []
        for _ in range(max(0, n)):
            if elites and self.rng.random() < 0.55:
                parent = self.rng.choice(elites).expression
                candidates.append(self._mutate_elite(parent))
            else:
                candidates.append(self._fresh_expression())

        self.usage.record(
            latency_seconds=perf_counter() - started,
            requested_proposals=max(0, n),
            returned_proposals=len(candidates),
            success=True,
        )
        return candidates

    def usage_summary(self) -> UsageSummary:
        return self.usage.summary()


def _read_field(value: object, name: str, default: object = 0) -> object:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_usage(response: object) -> tuple[int, int, int, int]:
    usage = _read_field(response, "usage", None)
    input_tokens = int(_read_field(usage, "input_tokens", 0) or 0)
    output_tokens = int(_read_field(usage, "output_tokens", 0) or 0)
    total_tokens = int(
        _read_field(usage, "total_tokens", input_tokens + output_tokens)
        or input_tokens + output_tokens
    )
    input_details = _read_field(usage, "input_tokens_details", None)
    cached_tokens = int(_read_field(input_details, "cached_tokens", 0) or 0)
    return input_tokens, cached_tokens, output_tokens, total_tokens


class OpenAIProposer:
    """Optional proposer backed by the OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        budget: ProposalBudget | None = None,
        pricing: TokenPricing | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be provided for the OpenAI proposer")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    'OpenAI support is optional. Install with: pip install -e ".[openai]"'
                ) from exc
            client = OpenAI()

        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.usage = UsageLedger(
            provider="openai",
            model=model,
            budget=budget,
            pricing=pricing,
        )

    @staticmethod
    def _prompt(elites: list[Elite], n: int) -> str:
        elite_lines = "\n".join(
            (
                f"- expression={elite.expression!r}; "
                f"fitness={elite.fitness:.8f}; "
                f"mean_excess_ratio={elite.mean_excess_ratio:.8f}; "
                f"W/T/L_vs_best_fit={elite.wins}/{elite.ties}/{elite.losses}"
            )
            for elite in elites
        )

        return f"""You are proposing interpretable priority heuristics for ONLINE 1D BIN PACKING.

For each arriving item, the expression is evaluated on every FEASIBLE open bin.
The bin with the HIGHEST score is selected.

Allowed variables:
item, capacity, remaining, residual, fill_ratio, item_ratio, tightness,
step_fraction, open_bins

Allowed functions: abs, min, max, sqrt, log1p, exp, tanh
Allowed constructs: arithmetic, comparisons, and conditional expressions.

Do not use imports, assignments, attributes, comprehensions, subscripts, lambdas,
randomness, state, files, network access, or any names outside the whitelist.

Elite candidates measured by the deterministic evaluator:
{elite_lines}

Generate {n} NEW candidate expressions. Prefer compact expressions that make
meaningful changes to the elites rather than cosmetic rewrites. Explore rules
that balance tight packing against avoiding unusable residual gaps.

Return ONLY a JSON array of strings.
"""

    @staticmethod
    def _parse_json_array(text: str) -> list[str]:
        cleaned = text.strip()
        fence = chr(96) * 3
        if cleaned.startswith(fence):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith(fence):
                lines = lines[1:]
            if lines and lines[-1].strip() == fence:
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end < start:
            raise ValueError("model output did not contain a JSON array")

        payload = json.loads(cleaned[start : end + 1])
        if not isinstance(payload, list):
            raise ValueError("model output must be a JSON array")

        return [
            value.strip()
            for value in payload
            if isinstance(value, str) and value.strip()
        ]

    def propose(self, elites: list[Elite], n: int) -> list[str]:
        self.usage.require_capacity()
        prompt = self._prompt(elites, n)
        request: dict[str, object] = {
            "model": self.model,
            "input": prompt,
            "store": False,
        }
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens

        started = perf_counter()
        response: object | None = None
        try:
            response = self.client.responses.create(**request)
            output_text = str(_read_field(response, "output_text", ""))
            candidates = self._parse_json_array(output_text)[:n]
            token_counts = _token_usage(response)
            self.usage.record(
                input_tokens=token_counts[0],
                cached_input_tokens=token_counts[1],
                output_tokens=token_counts[2],
                total_tokens=token_counts[3],
                latency_seconds=perf_counter() - started,
                requested_proposals=max(0, n),
                returned_proposals=len(candidates),
                success=True,
            )
            return candidates
        except Exception as exc:
            token_counts = _token_usage(response) if response is not None else (0, 0, 0, 0)
            self.usage.record(
                input_tokens=token_counts[0],
                cached_input_tokens=token_counts[1],
                output_tokens=token_counts[2],
                total_tokens=token_counts[3],
                latency_seconds=perf_counter() - started,
                requested_proposals=max(0, n),
                returned_proposals=0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def usage_summary(self) -> UsageSummary:
        return self.usage.summary()
