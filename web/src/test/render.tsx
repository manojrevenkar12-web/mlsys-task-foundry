import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { FUTURE, Routed } from "../app/App";

export type Handler = (url: string, init?: RequestInit) => { status?: number; body: unknown } | undefined;

/** Answers fetch from a table of handlers; anything unhandled is a 404 like the real API. */
export function mockApi(handler: Handler) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const r = handler(url, init);
    const status = r?.status ?? (r ? 200 : 404);
    const body = r ? r.body : { detail: `no route ${url}` };
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

export function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]} future={FUTURE}>
        <Routed />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}
