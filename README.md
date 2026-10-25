# mlsys-task-foundry

Tasks for training and evaluating models on ML-systems work (Pallas TPU kernels, profiler traces,
distributed JAX debugging and LLM serving), each with a grader that is **proven before anyone
solves the task**: it passes a reference solution and rejects every one of a set of plausible wrong
solutions, the mistakes engineers and models actually make.

![The proof matrix of the continuous-batching task: which check rejects which wrong solution](docs/img/proof.png)

## Why

Writing a hard task is the easy half. The hard half is a grader that cannot be fooled, and a reader
cannot tell from a grader's code which mistakes it would catch. Recent benchmark work shows how this
goes wrong: kernel graders that test one input distribution let shape-matched and hard-coded
solutions pass ([KernelBench-Verified](https://www.alphaxiv.org/abs/2607.16241)), and on JAX the
largest share of agent failures is calling APIs that no longer exist rather than getting the
algorithm wrong ([JAXBench](https://arxiv.org/pdf/2607.20466)). This repository treats a task's
wrong answers as part of the task.

## What is in it

22 tasks in four areas, all graded on CPU (see [what is simulated](#honest-limits)):

| Area | Tasks |
|---|---|
| Kernels (Pallas TPU, JAX 0.12) | tiled matmul with an f32 VMEM accumulator and correct dimension semantics for two TensorCores; row softmax that survives large logits and bf16; causal flash attention with online softmax; fused residual + RMSNorm with a ragged last block; int8 × int8 matmul with per-row and per-channel scales; porting a manual-DMA kernel from removed Pallas APIs |
| Profiling (traces) | a step that recompiles every batch (real `jax.profiler` trace, code fix plus answer); a host-bound input pipeline (real trace); exposed FSDP collectives on TPU (synthetic xprof-style trace); a launch-bound GPU optimizer (synthetic Kineto-style trace); roofline classification of LLM ops on an H100 |
| Debugging (8 simulated devices) | data-parallel replicas that drift apart after someone turned `check_vma` off; cross-entropy over a vocabulary sharded across devices; NaN gradients from an all-padding sequence; every shard drawing the same dropout mask; gradient accumulation that does not match the big batch |
| Serving | a paged KV-cache allocator with fork and copy-on-write; a continuous-batching scheduler with chunked prefill and preemption; prefix caching by chained block hashes; KV capacity planning for an 8B GQA model on an H100; when decode on a TPU v6e stops being memory-bound; speculative-decoding speedup and the best draft length |

Each task folder holds `task.yaml`, `prompt.md` (what the solver sees), sometimes `starter.py` and
`assets/`, `grader.py`, `reference/`, `mutants/<name>/` (each with the failure mode it represents)
and, for written answers, `rubric.yaml` with a pass and a fail example per criterion.

## Results

### The forge: every grader proven

`foundry forge` grades each reference twice and every mutant once
([reports/FORGE.md](reports/FORGE.md)):

- **22 of 22 tasks proven; 114 of 114 plausible wrong solutions rejected** by 105 checks
  (9 checks no mutant fails, listed per task as candidates for new mutants); 7.4 minutes on two CPU
  cores.
- Numeric checks run on five input distributions (uniform, ×100, ×0.01, negated, normal).
- Kernel references and starters are scanned for APIs removed or deprecated by JAX 0.12.0.1
  (`pltpu.TPUCompilerParams`, `pltpu.TPUMemorySpace`, `pl.dot`, `jax.experimental.shard_map`, …;
  a test confirms every name listed as removed in `config/api_deprecations.yaml` is really gone from
  the installed JAX).

The forge caught its author several times. The fixes are in the history:

- **A reference that was 8× wrong.** Under JAX 0.12's `shard_map` with `check_vma` on, the gradient
  of a loss with respect to a replicated parameter is already summed across the data axis by
  autodiff. Code written for older JAX then adds a `pmean`, which makes the update 8× too large on 8
  devices. The task's reference had done exactly that; the forge's comparison with a single-device
  run caught it. That mistake is now a mutant, and it is the point of the task.
- **Five mutants that were correct.** Two "NaN from padding" mutants masked with `-inf` and fixed
  the output with `where`; the textbook says the gradient goes NaN, but here the `where` on the scores
  stopped it before any parameter, so they were correct solutions. A scheduler "skip-ahead" mutant
  could never differ from the reference given its own invariants. Each was deleted or replaced with
  the mistake it meant to represent, never "killed" by making a grader reject correct code.
- **An ill-conditioned check**: flash attention on ×100 scaled queries and keys is ill-conditioned in
  float32 for any implementation; the check now bounds query and key magnitudes per distribution and
  then tests scores in the hundreds separately.
- **Rubric examples that disagreed with the trace**: pass examples written before the traces were
  recorded quoted numbers the traces do not give (5% busy instead of 11%). They now quote the trace.

### Blind attempts and a blind grader audit

BLIND_RESULTS

## How grading works

- **Checks** are plain functions in each `grader.py`, registered with
  `@check(targets=[failure modes], weight=…)`. They compute expected values independently: a float64
  NumPy reference for kernels, the same trace-reading functions the solver would use for profiling
  answers, an independent simulator that checks every scheduling decision for serving code.
- **The runner** grades in a child process with a nonce, a memory limit, a timeout, a temporary HOME
  and a stripped environment. A submission that prints a fake result line, patches `np.allclose` or
  the grader's comparison helpers, or hangs, gets a void or failed run (`tests/test_runner.py`).
- **Kernels** run in Pallas' TPU interpreter (`pltpu.InterpretParams`), which simulates VMEM, DMAs
  and semaphores, fills uninitialised memory with NaN and runs parallel grid points in random order
  on two simulated cores, so a reduction axis marked `parallel` fails instead of passing by luck.
- **Written answers** are graded twice: numbers and labels by checks, the explanation by people
  against the rubric in the console, with Cohen's kappa between two reviewers.

## The console

`foundry serve` serves a review console (React, TypeScript, Framer Motion) and its API:

- **Overview** and a task rail by area, with each task's proof status;
- **Task**: the prompt, starter and the failure modes it targets;
- **Proof**: the kill matrix, which check rejects which wrong solution, and anything idle;
- **Trace**: the task's profiler trace as lanes, with the longest totals;
- **Review**: the rubric with pass and fail examples, saved reviews and reviewer agreement;
- **Results**: blind attempts and audits, with what caught each failure.

Every screen has loading, error-with-retry and empty states.

| Overview | Trace | Review |
|---|---|---|
| ![](docs/img/overview.png) | ![](docs/img/trace.png) | ![](docs/img/review.png) |

## Run it

```bash
pip install -e ".[dev]"          # Python 3.11+, JAX 0.12.0.1 on CPU
foundry list                     # the tasks
foundry forge                    # prove every grader (about 7 minutes on 2 cores)
foundry grade tiled-matmul path/to/submission/
foundry attempt <label>          # grade attempts/<label>/<task>/, recorded once
foundry audit <label>            # grade audits/<label>/<task>/<variant>/, report escapes
cd web && npm ci && npm run build && cd .. && foundry serve
```

or `docker compose up` and open http://localhost:8000.

Tests: `pytest -q -m "not slow"` (seconds), `pytest -q -m slow` (forges every task), and
`cd web && npm test`. CI runs lint, mypy, both test sets, the console's typecheck, lint, tests and
build, and builds the image and checks it serves the API and the console.

## Honest limits

- **No accelerator was used.** Kernels are checked for correctness and memory-space discipline in the
  TPU interpreter, not for speed. Nothing here claims a kernel is fast.
- **Two of the four traces are synthetic.** The TPU collectives and GPU launch traces are generated
  by `scripts/make_assets.py` in the xprof and Kineto formats, because neither behaviour can be
  recorded on CPU. The recompilation and host-bound traces are real `jax.profiler` recordings.
- **Hardware numbers** (H100, TPU v5e and v6e) come from vendor pages cited in
  `config/hardware.yaml`; the serving tasks reason with them, they were not measured.
- **The blind solvers and auditors were models** working only from the solver's files, not human
  experts. The rubrics have not yet been through two independent human reviewers.
- **The runner isolates against a solver gaming the grader, not against malicious code.** Run
  untrusted submissions in a container or VM as well.

## Layout

```
config/          settings, hardware specs with sources, failure modes, API deprecations
src/foundry/     task loading, checks, runner, forge, API scan, traces, roofline, rubric, API
tasks/<area>/    the 22 tasks
scripts/         asset generation, mutant authoring
reports/         forge, attempt and audit reports (first runs are never rewritten)
attempts/ audits/ blind submissions
web/             the review console
docs/adr/        decisions: proven graders, CPU interpreters, isolation and first runs
```

MIT licence.
