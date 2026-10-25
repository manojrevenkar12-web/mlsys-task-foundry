# Data-parallel replicas that drift apart

`starter.py` trains a linear model with data parallelism over 8 devices: the weights are
replicated, the batch is split along `data`. Each step is supposed to be exactly a full-batch
gradient step on all `N` examples. It is not: the weights it returns depend on how many devices
there are, and someone disabled `shard_map`'s varying-across-devices check to silence an error.

Fix it in `solution.py`, keeping the interface (`make_mesh`, `loss_fn`, `make_step`, `train`).
After the fix:

- `train` returns the weights of `steps` full-batch gradient steps on the mean squared error over
  all `N` examples, and each step's loss is that full-batch loss, for any device count dividing `N`;
- the weights are truly identical on every device, and `make_step` runs with `shard_map`'s
  default checks on.

The grader runs on 8 simulated CPU devices (`--xla_force_host_platform_device_count=8`) and
compares with a single-device computation.
