"""Restricted arithmetic language for generated bin-scoring heuristics."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Mapping


ALLOWED_VARIABLES = {
    "item",
    "capacity",
    "remaining",
    "residual",
    "fill_ratio",
    "item_ratio",
    "tightness",
    "step_fraction",
    "open_bins",
}

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log1p": math.log1p,
    "exp": math.exp,
    "tanh": math.tanh,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.IfExp,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.BoolOp,
    ast.And,
    ast.Or,
)


class UnsafeExpression(ValueError):
    """Raised when a candidate uses syntax outside the restricted language."""


@dataclass(frozen=True, slots=True)
class CompiledExpression:
    source: str
    code: object

    def __call__(self, context: Mapping[str, float]) -> float:
        env = {name: float(context[name]) for name in ALLOWED_VARIABLES}
        env.update(ALLOWED_FUNCTIONS)
        try:
            value = eval(self.code, {"__builtins__": {}}, env)
        except (ArithmeticError, ValueError, OverflowError) as exc:
            raise ValueError(f"expression failed numerically: {self.source!r}") from exc
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"expression returned non-finite value: {result!r}")
        return result


class _Validator(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeExpression(f"unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ALLOWED_VARIABLES and node.id not in ALLOWED_FUNCTIONS:
            raise UnsafeExpression(f"unknown name: {node.id!r}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise UnsafeExpression("only whitelisted mathematical functions are allowed")
        if node.keywords:
            raise UnsafeExpression("keyword arguments are not allowed")
        if len(node.args) > 3:
            raise UnsafeExpression("function calls may have at most three arguments")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression("only numeric constants are allowed")
        if not math.isfinite(float(node.value)) or abs(float(node.value)) > 1_000:
            raise UnsafeExpression("numeric constant outside allowed range")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant):
                raise UnsafeExpression("power exponents must be numeric constants")
            exponent = float(node.right.value)
            if abs(exponent) > 4:
                raise UnsafeExpression("power exponent magnitude must be <= 4")
        self.generic_visit(node)


def compile_expression(source: str) -> CompiledExpression:
    """Parse and compile a candidate scoring expression after AST validation."""

    source = source.strip()
    if not source:
        raise UnsafeExpression("expression must not be empty")
    if len(source) > 500:
        raise UnsafeExpression("expression is too long")

    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"invalid expression syntax: {source!r}") from exc

    _Validator().visit(tree)
    code = compile(tree, filename="<heuristic>", mode="eval")
    return CompiledExpression(source=source, code=code)


def expression_complexity(source: str) -> int:
    """Count AST nodes as a small, transparent complexity proxy."""

    tree = ast.parse(source, mode="eval")
    return sum(1 for _ in ast.walk(tree))
