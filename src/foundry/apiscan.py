"""Find JAX and Pallas names that do not exist (any more) in the pinned version.

Models write Pallas from what they saw in training, and the API has moved fast: JAXBench found
API and runtime errors to be over half of agent failures. The scan resolves import aliases
(`from jax.experimental.pallas import tpu as pltpu`), so `pltpu.ANY`, a from-import of `ANY`, and
`import jax.experimental.maps` are all found, whatever the solution calls the module."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from . import config

SHORT = {
    "pl": "jax.experimental.pallas",
    "pltpu": "jax.experimental.pallas.tpu",
    "plgpu": "jax.experimental.pallas.triton",
}


@dataclass
class Finding:
    name: str
    line: int
    severity: str  # removed | deprecated
    note: str


def _expand(key: str) -> str:
    head, _, rest = key.partition(".")
    return f"{SHORT[head]}.{rest}" if head in SHORT else key


def catalogue() -> dict[str, tuple[str, str]]:
    d = config.load("api_deprecations")
    out: dict[str, tuple[str, str]] = {}
    for sev in ("removed", "deprecated"):
        for k, note in (d.get(sev) or {}).items():
            out[_expand(k)] = (sev, str(note))
    return out


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def scan(source: str) -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [Finding("<syntax>", e.lineno or 0, "removed", f"does not parse: {e.msg}")]
    cat = catalogue()
    alias: dict[str, str] = {}
    found: list[Finding] = []

    def hit(full: str, line: int) -> None:
        for name, (sev, note) in cat.items():
            if full == name or full.startswith(name + "."):
                found.append(Finding(name, line, sev, note))
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                hit(a.name, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                full = f"{node.module}.{a.name}"
                alias[a.asname or a.name] = full
                hit(full, node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            d = _dotted(node)
            if d:
                head, _, rest = d.partition(".")
                if head in alias:
                    hit(f"{alias[head]}.{rest}" if rest else alias[head], node.lineno)
    # one finding per name and line
    uniq = {(f.name, f.line): f for f in found}
    return sorted(uniq.values(), key=lambda f: (f.line, f.name))
