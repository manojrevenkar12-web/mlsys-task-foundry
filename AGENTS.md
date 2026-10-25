# Working on mlsys-task-foundry

## Fast feedback

- `pytest -q -m "not slow"`: runner, API, forge, rubric, trace and task-schema tests (seconds).
- `pytest -q -m slow`: forges every task (each reference twice, every mutant once; about 15 minutes on CPU).
- `foundry forge [--only a,b]`: the same, and writes `reports/forge.json` and `reports/FORGE.md`.
- `ruff check src tests scripts`, `ruff format --check src tests scripts`, `mypy src`.
- Console: `cd web && npm test && npm run typecheck && npm run lint && npm run build`.

## Invariants

- A task ships only when the forge proves it: the reference passes on every run, every mutant is
  rejected by at least one check, the reference and starter use current APIs, the prompt does not
  contain the solution, and the rubric lints. `require_all_mutants_killed` stays `true`.
- When a mutant survives, first ask whether it is actually wrong. An equivalent mutant is deleted
  (say so in the commit), not "fixed" by tightening the grader to reject correct code.
- Graders compute expected values from the task's own assets and an independent implementation;
  they never hard-code numbers that a changed asset would make stale.
- Submissions run in a child process with a nonce, a memory limit, a timeout, no network and a
  temporary HOME. Anything that patches the grader's helpers voids the run. Keep `_guarded()` in
  `runner.py` in step with any new helper a check depends on.
- Settings live in `config/` (`foundry.yaml`, `hardware.yaml`, `failure_modes.yaml`,
  `api_deprecations.yaml`), not in code. Task tolerances live in each `task.yaml`.
- Blind sets: `attempts/<label>/` and `audits/<label>/` are written by someone who saw only the
  solver's files (prompt, starter, assets). Their first runs in `reports/attempts/` and
  `reports/audits/` are never rewritten. Never change a grader to pass, or to catch, a recorded
  blind set; if you study its failures, mark it a dev set in its report's folder (`DEV`) and have
  someone who has not seen the graders write a new one.
- Examples are defensive and ordinary: no secrets, `.example` hosts, nothing that targets people.
