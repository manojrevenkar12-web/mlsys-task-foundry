import { motion as m } from "framer-motion";

import type { ForgeTask } from "../../shared/api/types";
import { humanise } from "../../shared/lib/format";
import { motion } from "../../shared/theme/motion";
import styles from "./ProofMatrix.module.css";

/** Every plausible wrong solution (rows) against every check (columns): a filled cell is a check
 * that rejects that solution. The reference row must stay empty, every other row must have at
 * least one filled cell. */
export function ProofMatrix({ forge, modeText }: { forge: ForgeTask; modeText: Record<string, string> }) {
  const checks = forge.checks;
  const rows = Object.entries(forge.mutants);
  let n = 0;
  const cell = (on: boolean, label: string) => {
    const i = n++;
    return (
      <m.span
        key={label}
        role="cell"
        aria-label={label}
        className={styles.cell}
        data-on={on || undefined}
        initial={{ opacity: 0, scale: 0.4 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ ...motion.spring, delay: i * motion.cellStagger }}
      />
    );
  };
  return (
    <div className={styles.wrap}>
      <div className={styles.verdict} data-ok={forge.ok || undefined}>
        <strong>{forge.ok ? "Proven" : "Not proven"}</strong>
        <span>
          The reference {forge.reference_passed ? "passes every check" : "fails"}
          {forge.deterministic ? " on every run" : ", and its results change between runs"} ({forge.reference_seconds < 1 ? "under a second" : `${forge.reference_seconds.toFixed(0)} s`}).{" "}
          {forge.killed} of {rows.length} plausible wrong solutions are rejected.
        </span>
        {forge.problems.length > 0 && (
          <ul className={styles.problems}>
            {forge.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}
      </div>

      <div className={styles.scroller} tabIndex={0} aria-label="Proof matrix, scroll sideways">
        <div role="table" aria-label="Which check rejects which wrong solution" className={styles.table} style={{ ["--cols" as string]: checks.length }}>
          <div role="row" className={styles.head}>
            <span role="columnheader" className={styles.rowLabel}>
              Solution
            </span>
            {checks.map((c, i) => (
              <span role="columnheader" key={c} className={styles.colHead} title={humanise(c)}>
                {i + 1}
              </span>
            ))}
            <span role="columnheader" className={`${styles.result} ${styles.resultHead}`}>
              Result
            </span>
          </div>
          <div role="row" className={styles.row} data-reference>
            <span role="rowheader" className={styles.rowLabel}>
              <span className={styles.name}>reference</span>
              <span className={styles.note}>The intended solution: no check may reject it.</span>
            </span>
            {checks.map((c) => cell(false, `reference passes ${humanise(c)}`))}
            <span role="cell" className={styles.result} data-good={forge.reference_passed || undefined}>
              {forge.reference_passed ? "passes" : "fails"}
            </span>
          </div>
          {rows.map(([name, r]) => {
            const rejected = r.killed_by.length > 0 || !!r.crashed;
            return (
              <div role="row" key={name} className={styles.row}>
                <span role="rowheader" className={styles.rowLabel}>
                  <span className={styles.name}>{name}</span>
                  <span className={styles.note}>{r.note}</span>
                </span>
                {checks.map((c) => cell(r.killed_by.includes(c), `${name}: ${r.killed_by.includes(c) ? "rejected by" : "passes"} ${humanise(c)}`))}
                <span role="cell" className={styles.result} data-good={rejected || undefined}>
                  {r.crashed && !r.killed_by.length ? "crashes" : rejected ? "rejected" : "escapes"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <ol className={styles.legend} aria-label="Checks, in the order they run">
        {checks.map((c) => (
          <li key={c}>
            <span className={styles.checkName}>{humanise(c)}</span>
            {forge.idle_checks.includes(c) && <span className={styles.idle}>no wrong solution fails it</span>}
            <span className={styles.targets}>
              {(forge.check_targets[c] ?? []).map((t) => (
                <span key={t} className={styles.mode} title={modeText[t] ?? t}>
                  {t}
                </span>
              ))}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
