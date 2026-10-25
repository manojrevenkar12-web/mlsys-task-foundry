import { Link } from "react-router-dom";

import { useAttempt, useAudit, useSummary } from "../../shared/api/queries";
import { AREA_LABEL, humanise } from "../../shared/lib/format";
import { EmptyState, QueryView } from "../../shared/ui/states";
import styles from "./ResultsPage.module.css";

function AttemptTable({ label }: { label: string }) {
  const q = useAttempt(label);
  return (
    <QueryView query={q} what={`attempt ${label}`} lines={6}>
      {(d) => (
        <section className={styles.block} aria-labelledby={`a-${label}`}>
          <h2 id={`a-${label}`} className={styles.h2}>
            {label}: {d.summary.passed} of {d.summary.attempted} tasks passed
          </h2>
          <p className={styles.sub}>
            Recorded {new Date(d.recorded_at).toLocaleString("en-GB")}
            {d.forced ? ", rewritten by hand (not a first run)" : ", first run, never rewritten"}.
          </p>
          <div className={styles.scroller}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Task</th>
                  <th scope="col">Area</th>
                  <th scope="col">Result</th>
                  <th scope="col">Caught by</th>
                </tr>
              </thead>
              <tbody>
                {d.tasks.map((t) => (
                  <tr key={t.task} data-passed={t.passed || undefined}>
                    <td>
                      <Link to={`/tasks/${t.task}`}>{t.task}</Link>
                    </td>
                    <td>{AREA_LABEL[t.area]}</td>
                    <td className={styles.result}>{t.passed ? "passed" : t.crashed ? "crashed" : `failed ${Math.round((1 - t.score) * 100)}% of checks`}</td>
                    <td className={styles.caught}>
                      {t.crashed && !t.failed_checks.length ? (
                        <span className={styles.message}>{t.crashed}</span>
                      ) : (
                        t.failed_checks.map((c) => (
                          <span key={c.name} title={c.message}>
                            {humanise(c.name)}
                          </span>
                        ))
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </QueryView>
  );
}

function AuditTable({ label }: { label: string }) {
  const q = useAudit(label);
  return (
    <QueryView query={q} what={`audit ${label}`} lines={6}>
      {(d) => (
        <section className={styles.block} aria-labelledby={`g-${label}`}>
          <h2 id={`g-${label}`} className={styles.h2}>
            Grader audit {label}: {d.summary.escaped} of {d.summary.variants} wrong submissions escaped
          </h2>
          <div className={styles.scroller}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Task</th>
                  <th scope="col">What the author made wrong</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {d.variants.map((v) => (
                  <tr key={`${v.task}/${v.variant}`} data-escaped={v.escaped || undefined}>
                    <td>
                      <Link to={`/tasks/${v.task}`}>{v.task}</Link>
                    </td>
                    <td>{v.claim}</td>
                    <td className={styles.result}>{v.escaped ? "escaped" : `caught: ${v.caught_by.map(humanise).join(", ")}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </QueryView>
  );
}

export function ResultsPage() {
  const s = useSummary();
  return (
    <main className={styles.page}>
      <header className={styles.intro}>
        <h1 className={styles.title}>Results</h1>
        <p className={styles.lede}>Attempts at the tasks by solvers that saw only the prompts, and audits of the graders by authors who never saw them.</p>
      </header>
      <QueryView
        query={s}
        what="results"
        lines={8}
        isEmpty={(d) => d.attempts.length === 0 && d.audits.length === 0}
        empty={
          <EmptyState title="No attempts or audits recorded">
            Put submissions in <code>attempts/&lt;label&gt;/&lt;task&gt;/</code> and run <code>foundry attempt &lt;label&gt;</code>.
          </EmptyState>
        }
      >
        {(d) => (
          <div className={styles.blocks}>
            {d.attempts.map((a) => (
              <AttemptTable key={a.label} label={a.label} />
            ))}
            {d.audits.map((a) => (
              <AuditTable key={a.label} label={a.label} />
            ))}
          </div>
        )}
      </QueryView>
    </main>
  );
}
