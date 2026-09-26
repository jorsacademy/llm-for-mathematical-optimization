"""Process-isolated evaluation for generated heuristic expressions."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from time import perf_counter

from .benchmarks import BenchmarkCase
from .evaluator import Evaluation, evaluate_expression
from .expressions import compile_expression


class SandboxEvaluationError(RuntimeError):
    """Raised when the isolated evaluator rejects or fails a candidate."""


class SandboxTimeout(SandboxEvaluationError):
    """Raised when candidate evaluation exceeds the configured wall-clock limit."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    evaluation: Evaluation
    wall_seconds: float


def _worker(
    connection: Connection,
    expression: str,
    suite: tuple[BenchmarkCase, ...],
    complexity_penalty: float,
) -> None:
    try:
        evaluation = evaluate_expression(
            expression,
            suite,
            complexity_penalty=complexity_penalty,
        )
        connection.send(("ok", evaluation))
    except Exception as exc:
        connection.send(("error", type(exc).__name__, str(exc)))
    finally:
        connection.close()


def evaluate_expression_isolated(
    expression: str,
    suite: tuple[BenchmarkCase, ...],
    *,
    timeout_seconds: float = 10.0,
    complexity_penalty: float = 1e-5,
) -> SandboxResult:
    """Validate then evaluate one candidate in a separate process.

    The expression DSL already forbids imports, attributes, subscripts, state,
    file/network primitives, and arbitrary function calls. This process boundary
    adds a wall-clock kill switch and keeps runtime failures out of the search
    process. It is not a general-purpose container for arbitrary Python code.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    # Fail unsafe syntax before process creation. The worker recompiles the same
    # expression, so no unchecked code crosses the process boundary.
    compile_expression(expression)

    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(send_connection, expression, suite, complexity_penalty),
        daemon=True,
    )

    started = perf_counter()
    process.start()
    send_connection.close()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        receive_connection.close()
        raise SandboxTimeout(
            f"candidate evaluation exceeded {timeout_seconds:.3f} seconds"
        )

    elapsed = perf_counter() - started
    if not receive_connection.poll():
        exit_code = process.exitcode
        receive_connection.close()
        raise SandboxEvaluationError(
            f"sandbox worker exited without a result (exit code {exit_code})"
        )

    message = receive_connection.recv()
    receive_connection.close()
    if message[0] == "error":
        _, error_type, error_message = message
        raise SandboxEvaluationError(f"{error_type}: {error_message}")

    _, evaluation = message
    if not isinstance(evaluation, Evaluation):
        raise SandboxEvaluationError("sandbox worker returned an unexpected payload")
    return SandboxResult(evaluation=evaluation, wall_seconds=elapsed)
