import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./client";
import type { Review } from "./types";

export const useSummary = () => useQuery({ queryKey: ["summary"], queryFn: api.summary });
export const useTasks = () => useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
export const useTask = (id: string | undefined) => useQuery({ queryKey: ["task", id], queryFn: () => api.task(id as string), enabled: !!id });
export const useTrace = (id: string, enabled: boolean) => useQuery({ queryKey: ["trace", id], queryFn: () => api.trace(id), enabled });
export const useAttempt = (label: string) => useQuery({ queryKey: ["attempt", label], queryFn: () => api.attempt(label) });
export const useAudit = (label: string) => useQuery({ queryKey: ["audit", label], queryFn: () => api.audit(label) });
export const useReviews = (task: string) => useQuery({ queryKey: ["reviews", task], queryFn: () => api.reviews(task) });

export function useAddReview(task: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (r: Omit<Review, "created_at" | "score">) => api.addReview(r),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["reviews", task] }),
  });
}
