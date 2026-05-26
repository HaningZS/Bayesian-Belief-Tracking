"""Final-answer extraction and normalization helpers."""

from __future__ import annotations

import re
import ast
import math
import operator
from collections.abc import Callable


_NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def extract_final_answer(text: str) -> str:
    """Extract a likely final answer from a generated reasoning trace."""

    stripped = text.strip()
    if not stripped:
        return ""

    if "####" in stripped:
        return stripped.rsplit("####", 1)[1].strip()

    boxed = _extract_last_boxed(stripped)
    if boxed is not None and boxed.strip():
        return boxed.strip()

    matches = _NUMBER_PATTERN.findall(stripped)
    if matches:
        return matches[-1].strip()
    return stripped.splitlines()[-1].strip()


def normalize_answer(answer: str) -> str:
    """Normalize common numeric answer formatting for exact-match scoring.

    For answers containing LaTeX commands (backslash), only whitespace is
    stripped so that expressions like ``\\frac{3}{4}`` survive intact.
    """

    value = extract_final_answer(answer) if "####" in answer or "\\boxed" in answer else answer
    value = value.strip()
    value = value.strip(".。")

    if "\\" in value:
        return re.sub(r"\s+", "", value)

    value = value.replace("$", "")
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<=\d)[A-Za-z%]+$", "", value)

    if re.fullmatch(r"[-+]?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def is_correct_answer(prediction_text: str, gold_answer: str, checker: str = "exact") -> bool:
    """Compare predicted and gold answers after extraction and normalization."""

    if checker not in {"exact", "symbolic"}:
        raise ValueError("checker must be 'exact' or 'symbolic'")
    raw_pred = _answer_for_comparison(prediction_text)
    raw_gold = _answer_for_comparison(gold_answer)
    pred = normalize_answer(raw_pred)
    gold = normalize_answer(raw_gold)
    if pred == gold:
        return True
    if "\\" in pred and "\\" in gold:
        if _latex_equal(pred, gold):
            return True
    if checker == "symbolic":
        return symbolic_answer_equal(raw_pred, raw_gold) or symbolic_answer_equal(pred, gold)
    return False


def symbolic_answer_equal(left: str, right: str, tolerance: float = 1e-9) -> bool:
    """Compare common numeric/LaTeX answer forms with a safe evaluator."""

    left_value = _parse_symbolic_answer(left)
    right_value = _parse_symbolic_answer(right)
    if left_value is None or right_value is None:
        return False
    return _symbolic_values_close(left_value, right_value, tolerance)


def _latex_equal(a: str, b: str) -> bool:
    """Whitespace-insensitive LaTeX comparison."""
    return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


def _answer_for_comparison(text: str) -> str:
    stripped = text.strip()
    if "####" in stripped or "\\boxed" in stripped:
        return extract_final_answer(stripped)
    if stripped.startswith("\\") or (stripped.startswith(("(", "[")) and ("\\" in stripped or "," in stripped)):
        return stripped
    return extract_final_answer(stripped)


def _parse_symbolic_answer(value: str):
    text = _strip_outer_wrappers(value.strip())
    parts = _split_top_level_commas(text)
    if len(parts) > 1:
        parsed_parts = [_parse_symbolic_answer(part) for part in parts]
        if any(part is None for part in parsed_parts):
            return None
        return parsed_parts
    expression = _latex_to_python_expression(text)
    try:
        parsed = ast.parse(expression, mode="eval")
        return float(_eval_expression(parsed.body))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


def _strip_outer_wrappers(value: str) -> str:
    text = value.strip().strip("$")
    text = text.replace("\\left", "").replace("\\right", "")
    while len(text) >= 2 and (
        (text[0] == "(" and text[-1] == ")") or (text[0] == "[" and text[-1] == "]")
    ):
        inner = text[1:-1]
        if _balanced(inner):
            text = inner.strip()
        else:
            break
    return text


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if parts:
        parts.append(value[start:].strip())
    return parts


def _balanced(value: str) -> bool:
    depth = 0
    for char in value:
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _latex_to_python_expression(value: str) -> str:
    expression = value.strip()
    expression = expression.replace("\\left", "").replace("\\right", "")
    expression = _replace_frac_commands(expression)
    expression = _replace_sqrt_commands(expression)
    expression = expression.replace("\\pi", "pi").replace("π", "pi")
    expression = expression.replace("^", "**")
    expression = expression.replace("{", "(").replace("}", ")")
    expression = re.sub(r"(?<=[0-9)])(?=(pi|sqrt|\())", "*", expression)
    expression = re.sub(r"(pi|\))(?=[0-9(])", r"\1*", expression)
    return expression


def _replace_frac_commands(value: str) -> str:
    return _replace_latex_command(value, "\\frac", 2, lambda groups: f"(({groups[0]})/({groups[1]}))")


def _replace_sqrt_commands(value: str) -> str:
    return _replace_latex_command(value, "\\sqrt", 1, lambda groups: f"sqrt({groups[0]})")


def _replace_latex_command(
    value: str,
    command: str,
    arity: int,
    render: Callable[[list[str]], str],
) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        if not value.startswith(command, index):
            output.append(value[index])
            index += 1
            continue
        cursor = index + len(command)
        groups: list[str] = []
        ok = True
        for _ in range(arity):
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor >= len(value) or value[cursor] != "{":
                ok = False
                break
            group, cursor = _read_braced_group(value, cursor)
            groups.append(_latex_to_python_expression(group))
        if not ok:
            output.append(command)
            index += len(command)
            continue
        output.append(render(groups))
        index = cursor
    return "".join(output)


def _read_braced_group(value: str, start: int) -> tuple[str, int]:
    depth = 0
    chars: list[str] = []
    index = start
    while index < len(value):
        char = value[index]
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    raise ValueError("unbalanced LaTeX braces")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expression(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return float(_BIN_OPS[type(node.op)](_eval_expression(node.left), _eval_expression(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_expression(node.operand)))
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        return math.sqrt(_eval_expression(node.args[0]))
    raise ValueError("unsupported symbolic expression")


def _symbolic_values_close(left, right, tolerance: float) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(_symbolic_values_close(a, b, tolerance) for a, b in zip(left, right, strict=True))
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return None
