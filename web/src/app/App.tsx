import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "framer-motion";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import { OverviewPage } from "../features/overview/OverviewPage";
import { ResultsPage } from "../features/results/ResultsPage";
import { TaskPage } from "../features/task/TaskPage";
import { ApiError } from "../shared/api/client";
import { EmptyState } from "../shared/ui/states";
import { Shell } from "./Shell";

const FUTURE = { v7_startTransition: true, v7_relativeSplatPath: true };

export function makeClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: (n, err) => !(err instanceof ApiError && err.status === 404) && n < 2, staleTime: 10_000 },
    },
  });
}

export function Routed() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/tasks/:id" element={<TaskPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="*" element={<EmptyState title="There is no page here">Pick a task on the left, or open Overview.</EmptyState>} />
      </Routes>
    </Shell>
  );
}

export function App({ client = makeClient() }: { client?: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <MotionConfig reducedMotion="user">
        <BrowserRouter future={FUTURE}>
          <Routed />
        </BrowserRouter>
      </MotionConfig>
    </QueryClientProvider>
  );
}

export { FUTURE };
