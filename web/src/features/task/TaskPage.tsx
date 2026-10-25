import { AnimatePresence, motion as m } from "framer-motion";
import { useParams, useSearchParams } from "react-router-dom";

import { useTask } from "../../shared/api/queries";
import type { TaskDetail } from "../../shared/api/types";
import { AREA_LABEL } from "../../shared/lib/format";
import { motion } from "../../shared/theme/motion";
import { EmptyState, QueryView } from "../../shared/ui/states";
import { PromptView } from "./PromptView";
import { ProofMatrix } from "./ProofMatrix";
import { ReviewPanel } from "./ReviewPanel";
import styles from "./TaskPage.module.css";
import { TraceView } from "./TraceView";

type Tab = "task" | "proof" | "trace" | "review";

function tabsFor(t: TaskDetail): { id: Tab; label: string }[] {
  const out: { id: Tab; label: string }[] = [
    { id: "task", label: "Task" },
    { id: "proof", label: "Proof" },
  ];
  if (t.assets.includes("trace.json.gz")) out.push({ id: "trace", label: "Trace" });
  if (t.rubric) out.push({ id: "review", label: "Review" });
  return out;
}

function Body({ task, tab }: { task: TaskDetail; tab: Tab }) {
  if (tab === "proof")
    return task.forge ? (
      <ProofMatrix forge={task.forge} modeText={task.mode_text} />
    ) : (
      <EmptyState title="Not forged yet">
        Run <code>foundry forge --only {task.id}</code> to prove its grader against the wrong solutions.
      </EmptyState>
    );
  if (tab === "trace") return <TraceView taskId={task.id} />;
  if (tab === "review") return <ReviewPanel task={task} />;
  return <PromptView task={task} />;
}

export function TaskPage() {
  const { id } = useParams();
  const [params, setParams] = useSearchParams();
  const q = useTask(id);
  return (
    <QueryView query={q} what="this task" lines={10}>
      {(task) => {
        const tabs = tabsFor(task);
        const wanted = params.get("tab") as Tab | null;
        const tab: Tab = tabs.some((t) => t.id === wanted) ? (wanted as Tab) : "task";
        return (
          <article className={styles.page} data-area={task.area}>
            <header className={styles.header}>
              <p className={styles.area}>
                {AREA_LABEL[task.area]}, difficulty {task.difficulty} of 5
              </p>
              <h1 className={styles.title}>{task.title}</h1>
            </header>
            <div className={styles.tabs} role="tablist" aria-label="Task views">
              {tabs.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  role="tab"
                  id={`tab-${t.id}`}
                  aria-selected={tab === t.id}
                  aria-controls="tab-panel"
                  className={styles.tab}
                  onClick={() => setParams(t.id === "task" ? {} : { tab: t.id }, { replace: true })}
                >
                  {t.label}
                  {tab === t.id && <m.span layoutId="tab-underline" className={styles.underline} transition={motion.spring} />}
                </button>
              ))}
            </div>
            <AnimatePresence mode="wait">
              <m.div
                key={tab}
                id="tab-panel"
                role="tabpanel"
                aria-labelledby={`tab-${tab}`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: motion.base, ease: motion.ease }}
              >
                <Body task={task} tab={tab} />
              </m.div>
            </AnimatePresence>
          </article>
        );
      }}
    </QueryView>
  );
}
