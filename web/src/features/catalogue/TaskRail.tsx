import { NavLink } from "react-router-dom";

import type { TaskRow } from "../../shared/api/types";
import { AREA_LABEL, AREAS } from "../../shared/lib/format";
import styles from "./TaskRail.module.css";

function Difficulty({ level }: { level: number }) {
  return (
    <span className={styles.difficulty} aria-label={`difficulty ${level} of 5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={styles.tick} data-on={i <= level || undefined} />
      ))}
    </span>
  );
}

function ProofMark({ t }: { t: TaskRow }) {
  if (!t.forge) return <span className={styles.proof} data-state="unproven">not forged</span>;
  const ok = t.forge.ok;
  return (
    <span className={styles.proof} data-state={ok ? "proven" : "broken"}>
      {ok ? `${t.forge.killed}/${t.forge.mutants} rejected` : "proof fails"}
    </span>
  );
}

export function TaskRail({ tasks }: { tasks: TaskRow[] }) {
  return (
    <nav aria-label="Tasks by area" className={styles.rail}>
      {AREAS.map((area) => {
        const mine = tasks.filter((t) => t.area === area);
        if (!mine.length) return null;
        return (
          <section key={area} className={styles.group} data-area={area} aria-labelledby={`rail-${area}`}>
            <h2 id={`rail-${area}`} className={styles.heading}>
              {AREA_LABEL[area]}
              <span className={styles.count}>{mine.length}</span>
            </h2>
            <ul className={styles.list}>
              {mine.map((t) => (
                <li key={t.id}>
                  <NavLink to={`/tasks/${t.id}`} className={styles.item}>
                    <span className={styles.title}>{t.title}</span>
                    <span className={styles.meta}>
                      <Difficulty level={t.difficulty} />
                      <ProofMark t={t} />
                    </span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </nav>
  );
}
