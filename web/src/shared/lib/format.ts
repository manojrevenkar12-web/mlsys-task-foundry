import type { Area } from "../api/types";

export const AREA_LABEL: Record<Area, string> = {
  kernels: "Kernels",
  profiling: "Profiling",
  debugging: "Debugging",
  serving: "Serving",
};

export const AREAS: Area[] = ["kernels", "profiling", "debugging", "serving"];

export const KIND_LABEL: Record<string, string> = {
  code: "Code",
  answer: "Written answer",
  "code+answer": "Code and written answer",
};

/** check_names_like_this -> "Check names like this" */
export function humanise(name: string): string {
  const s = name.replace(/[_-]+/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function pct(part: number, whole: number): string {
  return whole ? `${Math.round((part / whole) * 100)}%` : "–";
}

export function us(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)} ms`;
  return `${Math.round(v)} µs`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString("en")} ${n === 1 ? one : many}`;
}
