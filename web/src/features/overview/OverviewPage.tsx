import { Link } from "react-router-dom";

import { useSummary, useTasks } from "../../shared/api/queries";
import type { TaskRow } from "../../shared/api/types";
import { AREA_LABEL, AREAS } from "../../shared/lib/format";
import { EmptyState, QueryView } from "../../shared/ui/states";
import styles from "./OverviewPage.module.css";

function AreaRow({ area, tasks }: { area: TaskRow["area"]; tasks: TaskRow[] }) {
  const proven = tasks.filter((t) => t.forge?.ok).length;
  const mutants = tasks.reduce((n, t) => n + (t.forge?.mutants ?? 0), 0);
  const killed = tasks.reduce((n, t) => n + (t.forge?.killed ?? 0), 0);
  const first = tasks[0];
  const hard = tasks.filter((t) => t.tier === "hard").length;
  return (
    <tr data-area={area}>
      <th scope="row">{first ? <Link to={`/tasks/${first.id}`}>{AREA_LABEL[area]}</Link> : AREA_LABEL[area]}</th>
      <td>
        {tasks.length}
        {hard > 0 && <span className={styles.hard}> ({hard} hard)</span>}
      </td>
      <td>{proven}</td>
      <td>
        {killed} of {mutants}
      </td>
    </tr>
  );
}

export function OverviewPage() {
  const tasks = useTasks();
  const summary = useSummary();
  return (
    <main className={styles.page}>
      <header className={styles.intro}>
        <h1 className={styles.title}>Tasks whose graders are proven before anyone solves them</h1>
        <p className={styles.lede}>
          Each task comes with a reference solution and a set of plausible wrong ones: the mistakes engineers and models make on kernels,
          profiler traces, distributed code and LLM serving. A task counts as proven when its grader passes the reference and rejects every
          wrong solution.
        </p>
      </header>
      <QueryView query={tasks} what="tasks" lines={5} isEmpty={(d) => d.length === 0} empty={<EmptyState title="No tasks found">Add tasks under tasks/&lt;area&gt;/&lt;id&gt;/.</EmptyState>}>
        {(d) => (
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">Area</th>
                <th scope="col">Tasks</th>
                <th scope="col">Proven</th>
                <th scope="col">Wrong solutions rejected</th>
              </tr>
            </thead>
            <tbody>
              {AREAS.map((a) => (
                <AreaRow key={a} area={a} tasks={d.filter((t) => t.area === a)} />
              ))}
            </tbody>
          </table>
        )}
      </QueryView>
      <QueryView query={summary} what="results" lines={2}>
        {(s) =>
          s.attempts.length || s.audits.length ? (
            <p className={styles.results}>
              {s.attempts.map((a) => `${a.label}: ${a.passed} of ${a.attempted} tasks passed`).join(". ")}
              {s.audits.length ? `. Grader audit: ${s.audits.map((a) => `${a.escaped} of ${a.variants} wrong submissions escaped`).join(", ")}` : ""}.{" "}
              <Link to="/results">See results</Link>
            </p>
          ) : (
            <p className={styles.results}>No blind attempts recorded yet.</p>
          )
        }
      </QueryView>
    </main>
  );
}
