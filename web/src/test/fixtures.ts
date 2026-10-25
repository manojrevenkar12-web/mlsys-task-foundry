import type { AttemptDoc, AuditDoc, Reviews, Summary, TaskDetail, TaskRow } from "../shared/api/types";

export const rows: TaskRow[] = [
  { id: "tiled-matmul", title: "Tiled matmul in Pallas", area: "kernels", kind: "code", difficulty: 2, tier: "core", modes: ["wrong-tiling"], forge: { ok: true, killed: 3, mutants: 3, checks: 4 } },
  { id: "host-bound-input", title: "A host-bound input pipeline", area: "profiling", kind: "answer", difficulty: 3, tier: "hard", modes: ["host-bound"], forge: { ok: false, killed: 1, mutants: 2, checks: 3 } },
];

export const detail: TaskDetail = {
  id: "host-bound-input",
  title: "A host-bound input pipeline",
  area: "profiling",
  kind: "answer",
  difficulty: 3,
  tier: "hard",
  modes: ["host-bound"],
  target: "JAX profiler trace",
  skills: ["trace reading"],
  prompt: "# Why is the step slow?\n\nRead `assets/trace.json.gz`.",
  starter: null,
  rubric: {
    criteria: [
      { id: "numbers", weight: 60, text: "Cites the device busy share", pass_anchor: "says 11%", fail_anchor: "no number" },
      { id: "fix", weight: 40, text: "Proposes overlapping input with compute", pass_anchor: "prefetch", fail_anchor: "bigger batch" },
    ],
  },
  assets: ["trace.json.gz"],
  mode_text: { "host-bound": "The accelerator waits for the host." },
  forge: {
    id: "host-bound-input",
    reference_passed: true,
    deterministic: true,
    reference_seconds: 2.4,
    checks: ["busy_share", "names_decode", "overlap"],
    check_targets: { busy_share: ["host-bound"], names_decode: ["host-bound"], overlap: ["host-bound"] },
    mutants: {
      "span-based": { modes: ["host-bound"], note: "Measures from the first to the last event.", killed_by: ["busy_share"], crashed: null, seconds: 1 },
      "wrong-op": { modes: ["host-bound"], note: "Blames the optimizer.", killed_by: [], crashed: null, seconds: 1 },
    },
    idle_checks: ["overlap"],
    problems: ["mutant wrong-op passes every check"],
    ok: false,
    killed: 1,
  },
};

export const summary: Summary = {
  forge: { tasks: 2, ok: 1, mutants: 5, killed: 4, checks: 7, idle_checks: 1 },
  attempts: [{ label: "blind1", attempted: 2, passed: 1, not_attempted: 0, by_area: {} }],
  audits: [{ label: "audit1", variants: 3, escaped: 1 }],
};

export const emptySummary: Summary = { forge: null, attempts: [], audits: [] };

export const attempt: AttemptDoc = {
  label: "blind1",
  recorded_at: "2026-10-25T10:00:00+00:00",
  forced: false,
  tasks: [
    { task: "tiled-matmul", area: "kernels", difficulty: 2, passed: true, score: 1, crashed: null, failed_checks: [], modes: {} },
    {
      task: "host-bound-input",
      area: "profiling",
      difficulty: 3,
      passed: false,
      score: 0.5,
      crashed: null,
      failed_checks: [{ name: "busy_share", message: "busy share 0.9, want 0.113", targets: ["host-bound"] }],
      modes: { "host-bound": 1 },
    },
  ],
  missing: [],
  summary: { attempted: 2, passed: 1, failure_modes: { "host-bound": 1 } },
};

export const audit: AuditDoc = {
  label: "audit1",
  recorded_at: "2026-10-25T11:00:00+00:00",
  variants: [
    { task: "tiled-matmul", variant: "drops-last-tile", claim: "Ignores the last K tile.", escaped: false, caught_by: ["matches_reference"] },
    { task: "tiled-matmul", variant: "bf16-accumulate", claim: "Accumulates in bf16.", escaped: true, caught_by: [] },
  ],
  summary: { variants: 2, escaped: 1 },
};

export const noReviews: Reviews = { reviews: [], agreement: [] };
