import type { ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";

import { useTasks } from "../shared/api/queries";
import { QueryView } from "../shared/ui/states";
import { TaskRail } from "../features/catalogue/TaskRail";
import styles from "./Shell.module.css";

/** A 3x3 grid with the diagonal filled: the proof matrix, as a mark. */
function Mark() {
  return (
    <svg className={styles.mark} viewBox="0 0 12 12" aria-hidden="true">
      {[0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => <rect key={`${r}${c}`} x={c * 4} y={r * 4} width="3" height="3" rx="0.5" className={r === c ? styles.on : styles.off} />))}
    </svg>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const tasks = useTasks();
  return (
    <div className={styles.shell}>
      <a href="#content" className={styles.skip}>
        Skip to content
      </a>
      <header className={styles.bar}>
        <Link to="/" className={styles.brand}>
          <Mark />
          Task Foundry
        </Link>
        <nav aria-label="Sections" className={styles.nav}>
          <NavLink to="/" end className={styles.link}>
            Overview
          </NavLink>
          <NavLink to="/results" className={styles.link}>
            Results
          </NavLink>
        </nav>
      </header>
      <div className={styles.body}>
        <aside className={styles.rail}>
          <QueryView query={tasks} what="tasks" lines={12} isEmpty={(d) => d.length === 0}>
            {(d) => <TaskRail tasks={d} />}
          </QueryView>
        </aside>
        <div id="content" className={styles.content}>
          {children}
        </div>
      </div>
    </div>
  );
}
