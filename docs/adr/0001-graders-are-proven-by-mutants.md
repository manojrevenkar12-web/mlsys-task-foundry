# 1. A grader is proven by the wrong answers it rejects

## Context

A task for training or evaluating models on ML-systems work is only as good as its grader. Public
kernel benchmarks show the failure: graders that compare against one input distribution let
hard-coded or shape-matched solutions through (KernelBench-Verified found bypasses that pass by
special-casing the test shapes), and agents fail on deprecated APIs more than on the algorithm
(JAXBench). A reviewer reading a grader cannot tell which mistakes it would catch.

## Decision

Every task carries a reference solution and a set of mutants: realistic wrong solutions, each named
after a failure mode in `config/failure_modes.yaml`. The forge grades the reference twice and every
mutant once. A task is proven only when the reference passes both runs with the same results, every
mutant fails at least one check, the reference and starter use current APIs, the prompt does not
contain the reference's code, and the rubric lints. The forge also reports checks no mutant fails
(idle checks), which are candidates for new mutants.

Numeric checks run on five input distributions (uniform, scaled up ×100, scaled down ×0.01, negated,
normal) so a kernel that is only right on unit-scale inputs fails.

## Consequences

Writing a task means writing its wrong answers too, which roughly doubles the work and is the point:
the mutants are the record of what an expert expects to go wrong. The forge has rejected its own
author's work several times (an attention reference that was ill-conditioned on scaled inputs, a
data-parallel reference that was 8× off under JAX 0.12's automatic gradient psum, and seven mutants
that turned out to be correct solutions). Equivalent mutants are deleted, never "killed" by
tightening a grader against correct code.
