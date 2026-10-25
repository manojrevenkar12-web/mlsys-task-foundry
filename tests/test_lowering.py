"""lowers_for_tpu: the interpreter accepts kernels a TPU would reject; lowering (no TPU needed) does not."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

from foundry.checks import CheckFailed, lowers_for_tpu

SD = jax.ShapeDtypeStruct


def _double(x, block, interpret=None):
    def k(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2

    return pl.pallas_call(
        k,
        out_shape=SD(x.shape, x.dtype),
        grid=(x.shape[0] // block[0], x.shape[1] // block[1]),
        in_specs=[pl.BlockSpec(block, lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        interpret=interpret,
    )(x)


def test_a_tiled_block_lowers():
    lowers_for_tpu(lambda x: _double(x, (8, 128)), SD((64, 256), jnp.float32))


@pytest.mark.parametrize("block", [(4, 128), (8, 64)])
def test_a_block_off_the_tiling_runs_in_the_interpreter_but_does_not_lower(block):
    x = jnp.arange(64 * 256, dtype=jnp.float32).reshape(64, 256)
    assert jnp.array_equal(_double(x, block, interpret=True), x * 2)
    with pytest.raises(CheckFailed, match="does not lower for a TPU"):
        lowers_for_tpu(lambda a: _double(a, block), SD((64, 256), jnp.float32))


def test_interpret_mode_and_plain_xla_do_not_count():
    with pytest.raises(CheckFailed, match="without a Mosaic kernel"):
        lowers_for_tpu(lambda x: _double(x, (8, 128), interpret=True), SD((64, 256), jnp.float32))
    with pytest.raises(CheckFailed, match="without a Mosaic kernel"):
        lowers_for_tpu(lambda x: x * 2, SD((64, 256), jnp.float32))


def test_a_kernel_that_asks_which_chip_it_runs_on_still_lowers():
    from jax.experimental.pallas import tpu as pltpu

    def sized(x):
        rows = pltpu.get_tpu_info().num_sublanes  # 8 on every current TPU
        return _double(x, (rows, 128))

    lowers_for_tpu(sized, SD((64, 256), jnp.float32))
