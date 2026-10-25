"""Build the traces and tables the profiling tasks read. Real traces come from running the task's
own starter code under jax.profiler on CPU; the TPU and GPU traces are synthetic, generated from a
seeded model of a training step (they say so in their metadata and in the task prompts).

    python scripts/make_assets.py [names...]

Outputs are committed: a task's expected answers are computed from the committed file, so a
rerun on another machine changes nothing until someone commits the new trace."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"


def _write(path: Path, events: list[dict], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"displayTimeUnit": "ms", "traceEvents": events}
    if meta:
        doc["metadata"] = meta
    with gzip.open(path, "wt") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB, {len(events)} events)")


def _profile(fn, keep) -> list[dict]:
    import jax

    d = tempfile.mkdtemp()
    opts = jax.profiler.ProfileOptions()
    opts.python_tracer_level = 0
    opts.host_tracer_level = 2
    with jax.profiler.trace(d, create_perfetto_trace=True, profiler_options=opts):
        fn()
    p = next(Path(d).rglob("perfetto_trace.json.gz"))
    with gzip.open(p, "rt") as f:
        ev = json.load(f)["traceEvents"]
    return [e for e in ev if e.get("ph") == "M" or (e.get("ph") == "X" and keep(e))]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def recompiling() -> None:
    t = TASKS / "profiling" / "recompiling-train-step"
    starter = _load(t / "starter.py", "starter_recompile")
    import jax

    def run() -> None:
        params = starter.init(0)
        for i, (tokens, mask) in enumerate(starter.batches(0, 12)):
            with jax.profiler.StepTraceAnnotation("train", step_num=i):
                params, loss = starter.step(params, jax.numpy.asarray(tokens), jax.numpy.asarray(mask))
                loss.block_until_ready()

    keep_names = {
        "train",
        "PjitFunction(step)",
        "backend_compile_and_load",
        "lower_sharding_computation",
        "ThunkExecutor::Execute (wait for completion)",
    }
    events = _profile(run, lambda e: e["name"] in keep_names or ("jit_step" in e["name"] and e.get("dur", 0) > 200))
    _write(
        t / "assets" / "trace.json.gz", events, {"source": "jax.profiler on CPU, JAX 0.12.0.1, running starter.py for 12 batches"}
    )


def host_bound() -> None:
    t = TASKS / "profiling" / "host-bound-input"
    starter = _load(t / "starter.py", "starter_host")
    import jax

    def run() -> None:
        state = starter.init(0)
        for i in range(10):
            with jax.profiler.StepTraceAnnotation("train", step_num=i):
                state = starter.train_step(state, i)

    keep_names = {
        "train",
        "load_batch",
        "augment",
        "decode_images",
        "PjitFunction(update)",
        "ThunkExecutor::Execute (wait for completion)",
        "backend_compile_and_load",
    }
    events = _profile(run, lambda e: e["name"] in keep_names)
    _write(
        t / "assets" / "trace.json.gz", events, {"source": "jax.profiler on CPU, JAX 0.12.0.1, running starter.py for 10 steps"}
    )


def tpu_collectives(seed: int = 7) -> None:
    """A synthetic xprof-style trace of 8 FSDP training steps on one TPU core: a compute lane and
    two lanes of asynchronous collectives. Forward and backward all-gathers are prefetched one layer
    ahead and mostly hidden; each layer's gradient reduce-scatter starts when its backward ends, so
    the last one (layer 0) has no compute left to hide behind, and the optimizer waits for it."""
    import random

    rnd = random.Random(seed)
    t = TASKS / "profiling" / "tpu-exposed-collectives"
    pid = 1
    ev: list[dict] = [
        {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": "/device:TPU:0 (synthetic)"}},
        {"ph": "M", "pid": pid, "tid": 1, "name": "thread_name", "args": {"name": "Steps"}},
        {"ph": "M", "pid": pid, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
        {"ph": "M", "pid": pid, "tid": 3, "name": "thread_name", "args": {"name": "Async XLA Ops (gathers)"}},
        {"ph": "M", "pid": pid, "tid": 4, "name": "thread_name", "args": {"name": "Async XLA Ops (reductions)"}},
    ]

    def x(tid: int, name: str, ts: float, dur: float, **args: str) -> None:
        ev.append({"ph": "X", "pid": pid, "tid": tid, "name": name, "ts": round(ts, 3), "dur": round(dur, 3), "args": args})

    layers = 8
    ts = 0.0
    for step in range(8):
        s0 = ts
        # forward: layer 0's weights are gathered with nothing to overlap; then gather i+1 during i
        ag_end = ts + rnd.uniform(380, 420)
        x(3, "all-gather", ts, ag_end - ts, layer="0", phase="fwd")
        for i in range(layers):
            ts = max(ts, ag_end)  # compute needs this layer's weights
            fwd = rnd.uniform(520, 600)
            if i + 1 < layers:
                ag = rnd.uniform(380, 420)
                x(3, "all-gather", ts + 10, ag, layer=str(i + 1), phase="fwd")
                ag_end = ts + 10 + ag
            x(2, f"fusion.fwd.{i}", ts, fwd, layer=str(i))
            ts += fwd
        # backward, last layer first: its weights are gathered during the last forward layer
        ag_end = ts
        rs_free = ts
        for i in reversed(range(layers)):
            ts = max(ts, ag_end)
            bwd = rnd.uniform(1000, 1150)
            if i > 0:
                ag = rnd.uniform(380, 420)
                x(3, "all-gather", ts + 10, ag, layer=str(i - 1), phase="bwd")
                ag_end = ts + 10 + ag
            x(2, f"fusion.bwd.{i}", ts, bwd, layer=str(i))
            ts += bwd
            rs = rnd.uniform(700, 780)
            start = max(ts, rs_free)
            x(4, "reduce-scatter", start, rs, layer=str(i))
            rs_free = start + rs
        ts = max(ts, rs_free)  # the optimizer needs every gradient shard
        ar = rnd.uniform(40, 60)
        x(4, "all-reduce", ts, ar, what="loss and gradient norm")
        ts += ar
        opt = rnd.uniform(300, 340)
        x(2, "fusion.optimizer", ts, opt)
        ts += opt
        x(1, f"step {step}", s0, ts - s0, step_num=str(step))
        ts += rnd.uniform(20, 40)
    _write(
        t / "assets" / "trace.json.gz",
        ev,
        {"source": "synthetic, scripts/make_assets.py tpu_collectives(seed=7); xprof-style lanes"},
    )


def gpu_launch_bound(seed: int = 3) -> None:
    """A synthetic Kineto-style (torch.profiler) trace of 5 training iterations on one GPU: one
    large matmul, then an optimizer that launches hundreds of tiny element-wise kernels one by one."""
    import random

    rnd = random.Random(seed)
    t = TASKS / "profiling" / "gpu-launch-bound"
    cpu, gpu = 100, 0
    ev: list[dict] = [
        {"ph": "M", "pid": cpu, "name": "process_name", "args": {"name": "python (synthetic)"}},
        {"ph": "M", "pid": cpu, "tid": 1, "name": "thread_name", "args": {"name": "python main thread"}},
        {"ph": "M", "pid": gpu, "name": "process_name", "args": {"name": "GPU 0 (synthetic H100)"}},
        {"ph": "M", "pid": gpu, "tid": 7, "name": "thread_name", "args": {"name": "stream 7"}},
    ]
    ts = 0.0
    gpu_free = 0.0
    for it in range(5):
        i0 = ts
        kernels: list[tuple[str, float]] = [
            ("ampere_bf16_s16816gemm_128x256", rnd.uniform(380, 420)),
            ("ampere_bf16_s16816gemm_256x128", rnd.uniform(760, 840)),
        ]
        kernels += [("vectorized_elementwise_kernel<mul>", rnd.uniform(2.0, 3.5)) for _ in range(150)]
        kernels += [("vectorized_elementwise_kernel<add>", rnd.uniform(2.0, 3.5)) for _ in range(150)]
        kernels += [("vectorized_elementwise_kernel<lerp>", rnd.uniform(2.5, 4.0)) for _ in range(150)]
        kernels += [("vectorized_elementwise_kernel<sqrt_div>", rnd.uniform(3.0, 4.5)) for _ in range(150)]
        for name, dur in kernels:
            launch = rnd.uniform(6.5, 9.5)
            ev.append(
                {"ph": "X", "pid": cpu, "tid": 1, "name": "cudaLaunchKernel", "ts": ts, "dur": launch, "cat": "cuda_runtime"}
            )
            ts += launch
            start = max(ts + rnd.uniform(2.0, 4.0), gpu_free)
            ev.append({"ph": "X", "pid": gpu, "tid": 7, "name": name, "ts": start, "dur": dur, "cat": "kernel"})
            gpu_free = start + dur
            ts += rnd.uniform(0.5, 1.5)  # Python overhead between launches
        ts = max(ts, gpu_free) + 5
        ev.append(
            {"ph": "X", "pid": cpu, "tid": 1, "name": f"ProfilerStep#{it}", "ts": i0, "dur": ts - i0, "cat": "user_annotation"}
        )
    _write(
        t / "assets" / "trace.json.gz",
        ev,
        {"source": "synthetic, scripts/make_assets.py gpu_launch_bound(seed=3); Kineto-style categories"},
    )


def roofline_table() -> None:
    """Measured-looking op timings on an H100 (constructed: FLOPs and bytes are exact for each op's
    shape; times are chosen to land at a stated share of the roofline)."""
    import csv

    t = TASKS / "profiling" / "roofline-classify"
    rows = []

    def gemm(name: str, m: int, n: int, k: int, share: float) -> None:
        flops = 2 * m * n * k
        nbytes = (m * k + k * n + m * n) * 2
        rows.append((name, flops, nbytes, share))

    def stream(name: str, flops: float, nbytes: float, share: float) -> None:
        rows.append((name, flops, nbytes, share))

    gemm("prefill_gemm_4096x4096x4096", 4096, 4096, 4096, 0.84)
    gemm("prefill_gemm_512x4096x4096", 512, 4096, 4096, 0.70)
    gemm("prefill_gemm_256x4096x4096", 256, 4096, 4096, 0.75)
    gemm("decode_gemm_8x4096x4096", 8, 4096, 4096, 0.86)
    stream("decode_attention_b32_ctx2048", 2 * 2 * 32 * 2048 * 4096, 32 * 2048 * 2 * 4096 * 2, 0.80)
    stream("rmsnorm_8192x4096", 4 * 8192 * 4096, 2 * 8192 * 4096 * 2, 0.67)
    stream("rope_8192x4096", 6 * 8192 * 4096, 2 * 8192 * 4096 * 2 + 8192 * 128 * 4, 0.55)
    gemm("lm_head_64x128256x4096", 64, 128256, 4096, 0.81)

    sys.path.insert(0, str(ROOT / "src"))
    from foundry import roofline

    h100 = roofline.chip("h100_sxm")
    out = t / "assets" / "ops.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "flops", "bytes", "time_us"])
        for name, flops, nbytes, share in rows:
            seconds = flops / (roofline.attainable(h100, flops / nbytes) * share)
            w.writerow([name, int(flops), int(nbytes), round(seconds * 1e6, 3)])
    print(f"wrote {out}")


BUILDERS = {
    "roofline_table": roofline_table,
    "recompiling": recompiling,
    "host_bound": host_bound,
    "tpu_collectives": tpu_collectives,
    "gpu_launch_bound": gpu_launch_bound,
}


if __name__ == "__main__":
    for name in sys.argv[1:] or list(BUILDERS):
        BUILDERS[name]()
