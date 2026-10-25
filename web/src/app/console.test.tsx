import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { attempt, audit, detail, emptySummary, noReviews, rows, summary } from "../test/fixtures";
import { type Handler, mockApi, renderAt } from "../test/render";

const base: Handler = (url) => {
  if (url === "/api/tasks") return { body: rows };
  if (url === "/api/summary") return { body: summary };
  if (url === "/api/tasks/host-bound-input") return { body: detail };
  if (url.startsWith("/api/reviews")) return { body: noReviews };
  if (url === "/api/attempts/blind1") return { body: attempt };
  if (url === "/api/audits/audit1") return { body: audit };
  return undefined;
};

afterEach(() => vi.unstubAllGlobals());

describe("overview and rail", () => {
  it("shows a skeleton, then the areas and the task rail", async () => {
    mockApi(base);
    renderAt("/");
    expect(screen.getAllByRole("status")[0]).toHaveTextContent(/Loading tasks/);
    const rail = await screen.findByRole("navigation", { name: "Tasks by area" });
    expect(within(rail).getByRole("link", { name: /Tiled matmul in Pallas/ })).toHaveAttribute("href", "/tasks/tiled-matmul");
    expect(within(rail).getByRole("link", { name: /host-bound input pipeline/ })).toHaveTextContent(/Hard tier/);
    expect(within(rail).getByRole("link", { name: /Tiled matmul/ })).not.toHaveTextContent(/Hard tier/);
    const table = await screen.findByRole("table");
    const kernels = within(table).getByRole("row", { name: /Kernels/ });
    expect(kernels).toHaveTextContent("3 of 3");
    expect(await screen.findByText(/blind1: 1 of 2 tasks passed/)).toBeInTheDocument();
  });

  it("explains a failed load and recovers on retry", async () => {
    let fail = true;
    mockApi((url, init) => (url === "/api/tasks" && fail ? { status: 500, body: { detail: "forge.json is unreadable" } } : base(url, init)));
    renderAt("/");
    const [alert] = await screen.findAllByRole("alert");
    if (!alert) throw new Error("no alert");
    expect(alert).toHaveTextContent("Could not load tasks");
    expect(alert).toHaveTextContent("forge.json is unreadable");
    fail = false;
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("navigation", { name: "Tasks by area" })).toBeInTheDocument();
  });

  it("says so when there are no tasks", async () => {
    mockApi((url, init) => (url === "/api/tasks" ? { body: [] } : base(url, init)));
    renderAt("/");
    expect(await screen.findByText("No tasks found")).toBeInTheDocument();
  });

  it("has a page for addresses that lead nowhere", async () => {
    mockApi(base);
    renderAt("/nowhere");
    expect(await screen.findByText("There is no page here")).toBeInTheDocument();
  });
});

describe("a task", () => {
  it("opens on the prompt and switches tabs through the address", async () => {
    mockApi(base);
    renderAt("/tasks/host-bound-input");
    expect(await screen.findByRole("heading", { name: "Why is the step slow?" })).toBeInTheDocument();
    const tabs = screen.getByRole("tablist", { name: "Task views" });
    expect(within(tabs).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Task", "Proof", "Trace", "Review"]);
    fireEvent.click(within(tabs).getByRole("tab", { name: "Proof" }));
    expect(within(tabs).getByRole("tab", { name: "Proof" })).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText("Not proven")).toBeInTheDocument();
  });

  it("draws one cell per check and wrong solution, and flags the one that escapes", async () => {
    mockApi(base);
    renderAt("/tasks/host-bound-input?tab=proof");
    const matrix = await screen.findByRole("table", { name: "Which check rejects which wrong solution" });
    const body = within(matrix).getAllByRole("row").slice(1);
    expect(body).toHaveLength(3); // the reference and two wrong solutions
    expect(screen.getByText("mutant wrong-op passes every check")).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 plausible wrong solutions are rejected/)).toBeInTheDocument();
  });

  it("reports a task that does not exist", async () => {
    mockApi(base);
    renderAt("/tasks/gone");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/does not exist/);
  });

  it("saves a review only once every criterion is judged", async () => {
    const calls = mockApi((url, init) => {
      if (url === "/api/reviews" && init?.method === "POST") {
        const b = JSON.parse(String(init.body)) as { verdicts: Record<string, boolean> };
        return { status: 201, body: { ...b, created_at: "now", score: b.verdicts.numbers ? 60 : 0 } };
      }
      return base(url, init);
    });
    renderAt("/tasks/host-bound-input?tab=review");
    expect(await screen.findByText("No reviews yet")).toBeInTheDocument();
    const save = screen.getByRole("button", { name: "Save review" });
    expect(save).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "Ana" } });
    const first = screen.getByRole("radiogroup", { name: /device busy share/ });
    fireEvent.click(within(first).getByLabelText("Meets it"));
    expect(save).toBeDisabled();
    const second = screen.getByRole("radiogroup", { name: /overlapping input/ });
    fireEvent.click(within(second).getByLabelText("Does not"));
    expect(save).toBeEnabled();
    fireEvent.click(save);
    expect(await screen.findByText("Review saved: 60 of 100 points.")).toBeInTheDocument();
    const post = calls.find((c) => c.init?.method === "POST");
    expect(JSON.parse(String(post?.init?.body))).toEqual({
      task: "host-bound-input",
      submission: "reference",
      reviewer: "Ana",
      verdicts: { numbers: true, fix: false },
      notes: "",
    });
  });

  it("shows why a review was refused", async () => {
    mockApi((url, init) =>
      url === "/api/reviews" && init?.method === "POST" ? { status: 422, body: { detail: "verdicts must name this task's criteria" } } : base(url, init),
    );
    renderAt("/tasks/host-bound-input?tab=review");
    fireEvent.change(await screen.findByLabelText("Your name"), { target: { value: "Ana" } });
    for (const g of screen.getAllByRole("radiogroup")) fireEvent.click(within(g).getByLabelText("Meets it"));
    fireEvent.click(screen.getByRole("button", { name: "Save review" }));
    expect(await screen.findByText(/Could not save the review: verdicts must name/)).toBeInTheDocument();
  });
});

describe("results", () => {
  it("invites a first run when nothing is recorded", async () => {
    mockApi((url, init) => (url === "/api/summary" ? { body: emptySummary } : base(url, init)));
    renderAt("/results");
    expect(await screen.findByText(/foundry attempt/)).toBeInTheDocument();
  });

  it("lists attempts and audits with what caught each failure", async () => {
    mockApi(base);
    renderAt("/results");
    expect(await screen.findByRole("heading", { name: "blind1: 1 of 2 tasks passed" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /Grader audit audit1: 1 of 2 wrong submissions escaped/ })).toBeInTheDocument();
    expect(screen.getAllByText(/busy share/i).length).toBeGreaterThan(0);
    await waitFor(() => expect(screen.getByText("Accumulates in bf16.").closest("tr")).toHaveAttribute("data-escaped", "true"));
  });
});
