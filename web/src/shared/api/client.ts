import type { AttemptDoc, AuditDoc, Review, Reviews, Summary, TaskDetail, TaskRow, TraceDoc } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "The foundry API did not answer. Check that `foundry serve` is running.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "The request was not valid.";
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  summary: () => request<Summary>("/api/summary"),
  tasks: () => request<TaskRow[]>("/api/tasks"),
  task: (id: string) => request<TaskDetail>(`/api/tasks/${encodeURIComponent(id)}`),
  trace: (id: string) => request<TraceDoc>(`/api/tasks/${encodeURIComponent(id)}/trace`),
  attempt: (label: string) => request<AttemptDoc>(`/api/attempts/${encodeURIComponent(label)}`),
  audit: (label: string) => request<AuditDoc>(`/api/audits/${encodeURIComponent(label)}`),
  reviews: (task: string) => request<Reviews>(`/api/reviews?task=${encodeURIComponent(task)}`),
  addReview: (r: Omit<Review, "created_at" | "score">) =>
    request<Review>("/api/reviews", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(r) }),
};
