import importlib

import pytest

from foundry import apiscan


def test_finds_removed_names_through_any_alias():
    src = """
from jax.experimental import pallas as P
from jax.experimental.pallas import tpu as tpu_mod
from jax.experimental.pallas.tpu import TPUCompilerParams
import jax.experimental.maps

x = tpu_mod.ANY
y = P.dot(a, b)
z = P.ANY  # fine
"""
    found = {(f.name, f.line) for f in apiscan.scan(src)}
    assert ("jax.experimental.pallas.tpu.TPUCompilerParams", 4) in found
    assert ("jax.experimental.maps", 5) in found
    assert ("jax.experimental.pallas.tpu.ANY", 7) in found
    assert ("jax.experimental.pallas.dot", 8) in found
    assert not any(n.endswith("pallas.ANY") for n, _ in found)


def test_current_code_is_clean():
    src = "from jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\nx = pltpu.CompilerParams()\ny = pl.ANY\n"
    assert apiscan.scan(src) == []


def test_deprecated_is_reported_but_not_removed():
    f = apiscan.scan("from jax.experimental.shard_map import shard_map\n")
    assert f and f[0].severity == "deprecated"


def test_unparseable_source_is_a_finding():
    assert apiscan.scan("def (:\n")[0].name == "<syntax>"


@pytest.mark.parametrize("name", [n for n, (sev, _) in apiscan.catalogue().items() if sev == "removed"])
def test_every_removed_name_really_is_gone_from_the_pinned_jax(name):
    """The catalogue must not accuse names that still exist: each 'removed' entry fails to resolve."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return
            obj = getattr(obj, attr)
        pytest.fail(f"{name} still resolves in the installed JAX")
    # not importable at all: removed
