"""Strategy sandbox: execute LLM-generated code with limited surface area.

This is NOT a security sandbox. Anyone with access to the host can do anything.
The point is to prevent typical accidents:
    - Imports outside the whitelist (network, filesystem, exec).
    - Missing or wrong function signature.
    - Hangs from runaway computation (enforced via a wall-clock timeout).

The cleanest enforcement would be subprocess + seccomp. We use a constrained
`exec` namespace for portability. If you trust the model less than yourself,
swap this for `multiprocessing` + resource.setrlimit.
"""

from __future__ import annotations

import ast
import signal
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class SandboxError(RuntimeError):
    """Raised when sandboxed code fails for any reason."""


@dataclass(frozen=True)
class SandboxResult:
    positions: pd.Series | pd.DataFrame
    elapsed_seconds: float


_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "numpy",
        "np",
        "pandas",
        "pd",
        "math",
        "ai_quant_lab.features.library",
        "ai_quant_lab.features.cross_sectional",
    }
)


def run_strategy(
    source: str,
    price_data: pd.Series | pd.DataFrame,
    *,
    timeout_seconds: float = 10.0,
) -> SandboxResult:
    """Execute a strategy source string against price data.

    The source must define a function `strategy(price_data)`. Single-asset
    strategies accept a Series; cross-sectional strategies accept a DataFrame.
    The engine inspects the input type and validates the output shape matches.

    Args:
        source: Python source code defining `strategy`.
        price_data: Series (single asset) or DataFrame (multi-asset).
        timeout_seconds: Wall-clock cap; SIGALRM on POSIX.

    Returns:
        SandboxResult with the positions (Series or DataFrame matching input).

    Raises:
        SandboxError: on disallowed imports, missing function, exceptions,
            shape mismatch, or timeout.
    """
    _validate_imports(source)

    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "__builtins__": _safe_builtins(),
    }
    try:
        exec(compile(source, "<strategy>", "exec"), namespace)  # noqa: S102 — sandbox by design
    except Exception as exc:
        raise SandboxError(f"Strategy failed to compile: {exc}") from exc

    fn = namespace.get("strategy")
    if not callable(fn):
        raise SandboxError("Strategy source did not define a callable `strategy`.")

    import time

    start = time.perf_counter()
    with _alarm(timeout_seconds):
        try:
            positions = fn(price_data)
        except Exception as exc:
            raise SandboxError(f"Strategy raised: {exc}") from exc
    elapsed = time.perf_counter() - start

    expected_type = type(price_data).__name__
    if isinstance(price_data, pd.Series):
        if not isinstance(positions, pd.Series):
            raise SandboxError(
                f"Single-asset strategy must return a Series, got {type(positions).__name__}."
            )
        if not positions.index.equals(price_data.index):
            raise SandboxError("Strategy returned a Series with a different index.")
        return SandboxResult(positions=positions.fillna(0.0), elapsed_seconds=elapsed)

    if isinstance(price_data, pd.DataFrame):
        if not isinstance(positions, pd.DataFrame):
            raise SandboxError(
                f"Cross-sectional strategy must return a DataFrame, got {type(positions).__name__}."
            )
        if not positions.index.equals(price_data.index):
            raise SandboxError("Strategy returned a DataFrame with a different index.")
        if not positions.columns.equals(price_data.columns):
            raise SandboxError("Strategy returned a DataFrame with different columns.")
        return SandboxResult(positions=positions.fillna(0.0), elapsed_seconds=elapsed)

    raise SandboxError(f"Unsupported price_data type: {expected_type}")


def _validate_imports(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise SandboxError(f"Disallowed import: {name.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in _ALLOWED_IMPORTS:
                raise SandboxError(f"Disallowed import from: {mod}")


def _safe_builtins() -> dict[str, Any]:
    """Whitelist of builtins the strategy can call.

    We allow the usual math primitives but drop `open`, `eval`, `exec`, `__import__`.
    Numpy and pandas already give the strategy everything it needs.
    """
    import builtins

    safe = {
        name: getattr(builtins, name)
        for name in (
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
            "len", "list", "map", "max", "min", "range", "round", "set",
            "slice", "sorted", "str", "sum", "tuple", "type", "zip", "isinstance",
            "True", "False", "None",
        )
        if hasattr(builtins, name)
    }
    return safe


class _alarm:
    """Context manager that raises SandboxError if a timeout fires.

    Uses SIGALRM, which is POSIX-only. On non-POSIX or threaded contexts the
    timeout is silently a no-op.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._previous: Any = None
        self._active = False

    def __enter__(self) -> "_alarm":
        if hasattr(signal, "SIGALRM"):
            self._previous = signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous)

    @staticmethod
    def _handler(_signum: int, _frame: Any) -> None:
        raise SandboxError("Strategy exceeded the wall-clock timeout.")
