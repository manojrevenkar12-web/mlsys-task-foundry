import { AnimatePresence, motion as m } from "framer-motion";
import { type ReactNode, useState } from "react";
import { Link } from "react-router-dom";

import { useAttempt, useAudit, useSummary } from "../../shared/api/queries";
import type { AttemptTask, AuditDoc } from "../../shared/api/types";
import { AREA_LABEL, humanise, plural } from "../../shared/lib/format";
import { motion } from "../../shared/theme/motion";
import { EmptyState, QueryView } from "../../shared/ui/states";
import styles from "./ResultsPage.module.css";

function provenance(label: string, submissions: string, recorded: string, forced?: boolean) {
  const when = new Date(recorded).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  if (submissions !== label) return `Regrade of the ${submissions} submissions under the current graders, ${when}. The first run is ${submissions}.`;
  return forced ? `Recorded ${when}, rewritten by hand: not a first run.` : `Recorded ${when}. First run, never rewritten.`;
}

/** Rows that need attention always show; the rest open on request, so a long run stays scannable. */
function Rows<T>({ rows, notable, render, one, many }: { rows: T[]; notable: (r: T) => boolean; render: (r: T) => ReactNode; one: string; many: string }) {
  const [open, setOpen] = useState(false);
  const shown = rows.filter(notable);
  const rest = rows.filter((r) => !notable(r));
  return (
    <>
      <tbody>{shown.map(render)}</tbody>
      <AnimatePresence initial={false}>
        {open && (
          <m.tbody key="rest" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: motion.base }}>
            {rest.map(render)}
          </m.tbody>
        )}
      </AnimatePresence>
      {rest.length > 0 && (
        <tfoot>
          <tr>
            <td colSpan={4}>
              <button type="button" className={styles.more} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
                {`${open ? "Hide" : "Show"} ${plural(rest.length, one, many)}`}
              </button>
            </td>
          </tr>
        </tfoot>
      )}
    </>
  );
}

function AttemptRow(t: AttemptTask) {
  return (
    <tr key={t.task} data-good={t.passed || undefined}>
      <td>
        <Link to={`/tasks/${t.task}`}>{t.task}</Link>
      </td>
      <td>{AREA_LABEL[t.area]}</td>
      <td className={styles.result}>{t.passed ? "passed" : t.crashed ? "crashed" : `failed ${Math.round((1 - t.score) * 100)}% of checks`}</td>
      <td>
        <div className={styles.caught}>
          {t.crashed && !t.failed_checks.length ? (
            <span className={styles.message}>{t.crashed}</span>
          ) : (
            t.failed_checks.map((c) => (
              <span key={c.name} title={c.message}>
                {humanise(c.name)}
              </span>
            ))
          )}
        </div>
      </td>
    </tr>
  );
}

function AttemptTable({ label }: { label: string }) {
  const q = useAttempt(label);
  return (
    <QueryView query={q} what={`attempt ${label}`} lines={4}>
      {(d) => (
        <section className={styles.block} aria-labelledby={`a-${label}`}>
          <h2 id={`a-${label}`} className={styles.h2}>
            {label}: {d.summary.passed} of {d.summary.attempted} tasks passed
          </h2>
          <p className={styles.sub}>{provenance(d.label, d.submissions, d.recorded_at, d.forced)}</p>
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
              <Rows rows={d.tasks} notable={(t) => !t.passed} render={AttemptRow} one="passed task" many="passed tasks" />
            </table>
          </div>
        </section>
      )}
    </QueryView>
  );
}

function AuditRow(v: AuditDoc["variants"][number]) {
  return (
    <tr key={`${v.task}/${v.variant}`} data-good={!v.escaped || undefined} data-escaped={v.escaped || undefined}>
      <td>
        <Link to={`/tasks/${v.task}`}>{v.task}</Link>
      </td>
      <td>{v.claim}</td>
      <td className={styles.result}>{v.escaped ? "escaped" : `caught: ${v.caught_by.map(humanise).join(", ")}`}</td>
    </tr>
  );
}

function AuditTable({ label }: { label: string }) {
  const q = useAudit(label);
  return (
    <QueryView query={q} what={`audit ${label}`} lines={4}>
      {(d) => (
        <section className={styles.block} aria-labelledby={`g-${label}`}>
          <h2 id={`g-${label}`} className={styles.h2}>
            Grader audit {label}: {d.summary.escaped} of {d.summary.variants} wrong submissions escaped
          </h2>
          <p className={styles.sub}>{provenance(d.label, d.submissions, d.recorded_at)}</p>
          <div className={styles.scroller}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Task</th>
                  <th scope="col">What the author made wrong</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <Rows rows={d.variants} notable={(v) => v.escaped} render={AuditRow} one="caught submission" many="caught submissions" />
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
        <p className={styles.lede}>
          Attempts at the tasks by solvers that saw only the prompts, and audits of the graders by authors who never saw them. First runs come
          first; regrades show the same submissions under today's graders.
        </p>
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
