export type Area = "kernels" | "profiling" | "debugging" | "serving";

export interface ForgeBrief {
  ok: boolean | null;
  killed: number | null;
  mutants: number;
  checks: number;
}

export interface TaskRow {
  id: string;
  title: string;
  area: Area;
  kind: "code" | "answer" | "code+answer";
  difficulty: number;
  tier: "core" | "hard";
  modes: string[];
  forge: ForgeBrief | null;
}

export interface MutantResult {
  modes: string[];
  note: string;
  killed_by: string[];
  crashed: string | null;
  seconds: number;
}

export interface ForgeTask {
  id: string;
  reference_passed: boolean;
  deterministic: boolean;
  reference_seconds: number;
  checks: string[];
  check_targets: Record<string, string[]>;
  mutants: Record<string, MutantResult>;
  idle_checks: string[];
  problems: string[];
  ok: boolean;
  killed: number;
}

export interface Criterion {
  id: string;
  weight: number;
  text: string;
  pass_anchor: string;
  fail_anchor: string;
}

export interface TaskDetail extends Omit<TaskRow, "forge"> {
  target: string;
  skills: string[];
  prompt: string;
  starter: string | null;
  rubric: { criteria: Criterion[] } | null;
  assets: string[];
  forge: ForgeTask | null;
  mode_text: Record<string, string>;
}

export interface Summary {
  forge: { tasks: number; ok: number; mutants: number; killed: number; checks: number; idle_checks: number } | null;
  attempts: { label: string; submissions: string; regrade: boolean; attempted: number; passed: number; not_attempted: number; by_area: Record<string, { passed: number; of: number }> }[];
  audits: { label: string; submissions: string; regrade: boolean; variants: number; escaped: number }[];
}

export interface AttemptTask {
  task: string;
  area: Area;
  difficulty: number;
  passed: boolean;
  score: number;
  crashed: string | null;
  failed_checks: { name: string; message: string; targets: string[] }[];
  modes: Record<string, number>;
}

export interface AttemptDoc {
  label: string;
  submissions: string;
  recorded_at: string;
  forced: boolean;
  tasks: AttemptTask[];
  missing: string[];
  summary: { attempted: number; passed: number; failure_modes: Record<string, number> };
}

export interface AuditDoc {
  label: string;
  submissions: string;
  recorded_at: string;
  variants: { task: string; variant: string; claim: string; escaped: boolean; caught_by: string[] }[];
  summary: { variants: number; escaped: number };
}

export interface TraceDoc {
  start_us: number;
  end_us: number;
  lanes: { name: string; events: { name: string; ts: number; dur: number }[] }[];
  summary: { process: string; thread: string; name: string; total_us: number; count: number }[];
}

export interface Review {
  task: string;
  submission: string;
  reviewer: string;
  verdicts: Record<string, boolean>;
  notes: string;
  created_at: string;
  score?: number;
}

export interface Reviews {
  reviews: Review[];
  agreement: { task: string; submission: string; reviewers: string[]; kappa: number | null; criteria: number }[];
}
