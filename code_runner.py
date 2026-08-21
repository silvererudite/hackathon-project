#!/usr/bin/env python3
"""Restricted one-shot Python executor used by the code-generating agent."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from plotting import plot_corr_matrix

HERE = Path(__file__).resolve().parent
DENIED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals", "help",
    "input", "locals", "open", "os", "pathlib", "shutil", "socket", "subprocess",
    "sys", "vars",
}
DENIED_ATTRIBUTES = {
    "dump", "dumps", "load", "loads", "popen", "read_csv", "read_json",
    "read_pickle", "save", "savetxt", "system", "to_csv", "to_excel", "to_json",
    "to_pickle",
}


class SafetyError(ValueError):
    pass


def normalise_code(code: str) -> str:
    code = str(code or "")
    fenced = code.strip()
    if fenced.startswith("```") and fenced.endswith("```"):
        lines = fenced.splitlines()
        if len(lines) >= 2:
            code = "\n".join(lines[1:-1])
    code = textwrap.dedent(code).strip()
    lines = code.splitlines()
    for _ in range(8):
        try:
            ast.parse("\n".join(lines), mode="exec")
            break
        except IndentationError as exc:
            index = (exc.lineno or 0) - 1
            if "unexpected indent" not in str(exc) or not 0 <= index < len(lines):
                break
            indentation = len(lines[index]) - len(lines[index].lstrip())
            if not 0 < indentation < 4:
                break
            lines[index] = lines[index].lstrip()
    return "\n".join(lines)


def validate(code: str) -> None:
    if len(code) > 20_000:
        raise SafetyError("Generated code exceeds the 20,000 character limit.")
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise SafetyError(f"{type(node).__name__} statements are not allowed.")
        if isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            raise SafetyError(f"Name {node.id!r} is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENIED_ATTRIBUTES:
                raise SafetyError(f"Attribute {node.attr!r} is not allowed.")
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assignments.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            assignments.append(node.target)
    if not any(isinstance(target, ast.Name) and target.id == "result" for target in assignments):
        raise SafetyError("Code must assign its JSON-serializable answer to `result`.")


def load_data() -> pd.DataFrame:
    return pd.read_csv(HERE / "correlates_common_subjects.csv")


def serialise(value):
    if isinstance(value, pd.DataFrame):
        return {"shape": list(value.shape), "columns": list(value.columns),
                "rows": value.head(100).replace({np.nan: None}).to_dict(orient="records")}
    if isinstance(value, pd.Series):
        return value.head(200).replace({np.nan: None}).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialise(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def execute(code: str, history: list | None = None,
            figure_path: str | None = None) -> dict:
    code = normalise_code(code)
    validate(code)
    history = history or []
    for previous in history:
        previous_code = previous["code"] if isinstance(previous, dict) else previous
        validate(normalise_code(previous_code))
    correlates = load_data()
    safe_builtins = {
        "__import__": __import__,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "Exception": Exception,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "print": print, "range": range, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    }
    namespace = {
        "__builtins__": safe_builtins,
        "np": np, "pd": pd, "stats": stats,
        "pearsonr": pearsonr, "spearmanr": spearmanr,
        "correlates": correlates,
        "plot_corr_matrix": plot_corr_matrix,
        "figure_path": figure_path,
    }
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        for previous in history:
            previous_code = normalise_code(
                previous["code"] if isinstance(previous, dict) else previous)
            if isinstance(previous, dict):
                namespace["figure_path"] = previous.get("figure_path")
            exec(compile(previous_code, "<prior-generated-analysis>", "exec"),
                 namespace, namespace)
        namespace["figure_path"] = figure_path
        exec(compile(code, "<generated-analysis>", "exec"), namespace, namespace)
    artifacts = []
    if figure_path and Path(figure_path).is_file():
        artifacts.append({"type": "image", "path": str(Path(figure_path).resolve())})
    return {"result": serialise(namespace["result"]),
            "stdout": stream.getvalue()[-4000:], "artifacts": artifacts}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        response = {"ok": True, **execute(
            str(request.get("code", "")), list(request.get("history") or []),
            request.get("figure_path"))}
    except Exception as exc:
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
