# The same dropout mask on every shard

`starter.py` applies dropout inside a data-parallel computation over 8 devices. Training works, but
worse than on one device, and a colleague noticed that rows 0, B/8, 2B/8, ... of the batch always
lose exactly the same features, step after step.

Fix it in `solution.py`, keeping `make_mesh`, `dropout`, `make_apply` and `dropped`:

- every shard of the batch gets its own dropout mask, and every step gets a new one;
- the masks are fully determined by `key` and `step`: the same `(key, step)` gives the same result,
  on the same number of devices;
- the keep probability stays `1 - rate`, and kept values are scaled by `1 / (1 - rate)`.
