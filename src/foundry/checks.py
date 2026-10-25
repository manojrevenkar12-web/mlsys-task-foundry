"""What graders are written against.

    from foundry.checks import check, Ctx

    @check(targets=["reduction-axis-parallel"])
    def matches_reference_on_every_distribution(sol, ctx: Ctx):
        for name, (a, b) in ctx.inputs(lambda make: (make((256, 512)), make((512, 256)))):
            ctx.close(sol.matmul(a, b), a @ b, what=f"matmul on {name} inputs")

A check passes by returning and fails by raising `CheckFailed` (or AssertionError). Checks run in
the order they are defined. `targets` names the failure modes (config/failure_modes.yaml) a check
is there to catch; the forge reports which checks kill which mutants against those claims.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CHECKS_ATTR = "__foundry_check__"


class CheckFailed(AssertionError):
    pass


@dataclass
class CheckSpec:
    name: str
    targets: list[str]
    weight: float
    doc: str
    fn: Callable[..., Any]


def check(targets: list[str] | None = None, weight: float = 1.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, CHECKS_ATTR, CheckSpec(fn.__name__, list(targets or []), weight, (fn.__doc__ or "").strip(), fn))
        return fn

    return wrap


def collect(module: Any) -> list[CheckSpec]:
    specs = [getattr(v, CHECKS_ATTR) for v in vars(module).values() if hasattr(v, CHECKS_ATTR)]
    order = {name: i for i, name in enumerate(vars(module))}
    return sorted(specs, key=lambda s: order[s.name])


@dataclass
class Ctx:
    """Everything a check may use: the answer (answer tasks), assets, seeded randomness, the
    hidden input distributions and tolerant comparison."""

    task_dir: Path
    answer: dict[str, Any] = field(default_factory=dict)
    distributions: list[str] = field(default_factory=lambda: ["uniform", "scaled_up", "scaled_down", "negated", "normal"])
    scale_up: float = 100.0
    scale_down: float = 0.01
    seed: int = 0

    @property
    def assets(self) -> Path:
        return self.task_dir / "assets"

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1_000_003 + salt)

    def make(self, dist: str, salt: int = 0) -> Callable[..., np.ndarray]:
        """A factory for arrays from one distribution: make(shape, dtype=np.float32)."""
        rng = self.rng(salt)

        def mk(shape: tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
            base = rng.random(shape)
            x = {
                "uniform": base,
                "scaled_up": base * self.scale_up,
                "scaled_down": base * self.scale_down,
                "negated": -base,
                "normal": rng.standard_normal(shape),
            }[dist]
            return np.asarray(x, dtype=dtype)

        return mk

    def inputs(self, build: Callable[[Callable[..., np.ndarray]], Any]) -> Iterator[tuple[str, Any]]:
        """Build the same inputs once per hidden distribution."""
        for i, dist in enumerate(self.distributions):
            yield dist, build(self.make(dist, salt=i))

    def close(self, got: Any, want: Any, rtol: float = 1e-5, atol: float = 1e-5, what: str = "result") -> None:
        g = np.asarray(got, dtype=np.float64)
        w = np.asarray(want, dtype=np.float64)
        if g.shape != w.shape:
            raise CheckFailed(f"{what}: shape {g.shape}, expected {w.shape}")
        if not np.all(np.isfinite(g) == np.isfinite(w)):
            raise CheckFailed(f"{what}: {int(np.sum(~np.isfinite(g)))} non-finite values where the reference is finite")
        ok = np.isclose(g, w, rtol=rtol, atol=atol) | (~np.isfinite(w) & (g == w))
        if not np.all(ok):
            i = np.unravel_index(np.argmax(np.abs(g - w) * ~ok), g.shape)
            raise CheckFailed(
                f"{what}: {int(np.sum(~ok))} of {ok.size} values off; worst at {tuple(int(x) for x in i)}: got {g[i]:.6g}, expected {w[i]:.6g}"
            )

    def number(self, key: str, want: float, rel: float = 0.02, abs_: float = 0.0) -> None:
        """An answer field that must be within `rel` (relative) or `abs_` of `want`."""
        if key not in self.answer:
            raise CheckFailed(f"answer has no {key!r}")
        try:
            got = float(self.answer[key])
        except (TypeError, ValueError) as e:
            raise CheckFailed(f"{key}: {self.answer[key]!r} is not a number") from e
        if not (abs(got - want) <= max(abs_, rel * abs(want))):
            raise CheckFailed(f"{key}: {got:g} is not within {rel:.0%} of the correct value")

    def label(self, key: str, allowed: set[str] | str) -> None:
        """An answer field that must be one of the accepted labels (case-insensitive)."""
        want = {allowed} if isinstance(allowed, str) else allowed
        got = str(self.answer.get(key, "")).strip().lower()
        if got not in {w.lower() for w in want}:
            raise CheckFailed(f"{key}: {got!r} is not the right answer")


def fail(msg: str) -> None:
    raise CheckFailed(msg)
