# 3. Submissions run isolated, and a blind first run is recorded once

## Context

Models trained against a grader learn to attack it: print a fake result, patch the comparison
function, read the reference from disk, or loop forever. And a team that tunes its graders on the
same attempts it reports has measured nothing.

## Decision

The runner grades in a child process that receives a nonce on stdin and must print exactly one
result line carrying it. It limits memory (RLIMIT_AS) and time, points HOME at a temporary folder, and
strips the environment to a handful of variables (no proxy settings, so no network where egress
goes through a proxy; this is not a network sandbox). It checks before and after every check that the grader's helpers
(`Ctx.close`, `np.isclose`, `np.allclose`, `fail`) are the same objects; a patched helper voids the
run.

Blind attempts and audits are written by someone who saw only the solver's files. Their first run is
written to `reports/attempts/<label>.json` or `reports/audits/<label>.json` once; `foundry attempt`
refuses to overwrite it, and `--force` exists only for dev sets and is recorded in the report.

## Consequences

Grading costs a process start per submission (about a second with JAX imported). Escapes found by an
audit are fixed with new checks and new mutants, and the next audit is written by someone new.

The isolation is against a solver gaming the grader, not against malicious code: run untrusted
submissions inside a container or VM as well.
