# Causal ring attention

Write `ring_attention(q, k, v, *, axis_name)` in `solution.py`. It runs on every device of a
one-dimensional mesh inside `jax.shard_map`, with the sequence split evenly across the devices:
device `i` of `n` holds positions `[i * S_local, (i + 1) * S_local)` of `q`, `k` and `v`, each of
shape `(S_local, H, D)`. It returns that device's `(S_local, H, D)` slice of causal softmax
attention, `softmax(q kᵀ / √D)` with each position attending to itself and every earlier position
of the whole sequence, in the inputs' dtype.

Requirements:

- K and V travel between devices only with `jax.lax.ppermute`: no device may gather the whole
  sequence (no `all_gather`, `all_to_all`, `psum` or similar).
- The result and its gradients with respect to `q`, `k` and `v` (through `jax.grad` of the
  shard-mapped function) must match single-device causal attention, and be finite.
- The grader calls it exactly like this, on 8 CPU devices, with JAX 0.12.0.1 and shard_map's
  default checks:

```python
mesh = jax.make_mesh((8,), ("seq",), axis_types=(jax.sharding.AxisType.Auto,))
f = jax.jit(jax.shard_map(functools.partial(ring_attention, axis_name="seq"), mesh=mesh,
                          in_specs=(P("seq"), P("seq"), P("seq")), out_specs=P("seq")))
```

Inputs are float32 or bfloat16 (compute in float32). `S_local` may be any positive size.
