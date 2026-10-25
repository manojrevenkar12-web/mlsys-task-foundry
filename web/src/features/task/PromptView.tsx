import type { TaskDetail } from "../../shared/api/types";
import { KIND_LABEL } from "../../shared/lib/format";
import { Markdown } from "../../shared/ui/Markdown";
import styles from "./PromptView.module.css";

export function PromptView({ task }: { task: TaskDetail }) {
  return (
    <div className={styles.grid}>
      <article className={styles.prompt}>
        <Markdown text={task.prompt} />
        {task.starter && (
          <section aria-labelledby="starter-h" className={styles.starter}>
            <h3 id="starter-h" className={styles.h3}>
              starter.py
            </h3>
            <pre>
              <code>{task.starter}</code>
            </pre>
          </section>
        )}
      </article>
      <aside className={styles.facts} aria-label="About this task">
        <dl>
          <dt>Answer</dt>
          <dd>{KIND_LABEL[task.kind] ?? task.kind}</dd>
          <dt>Runs on</dt>
          <dd>{task.target}</dd>
          {task.assets.length > 0 && (
            <>
              <dt>Files</dt>
              <dd>
                {task.assets.map((a) => (
                  <code key={a} className={styles.asset}>
                    assets/{a}
                  </code>
                ))}
              </dd>
            </>
          )}
          <dt>Skills</dt>
          <dd className={styles.chips}>
            {task.skills.map((s) => (
              <span key={s} className={styles.chip}>
                {s}
              </span>
            ))}
          </dd>
        </dl>
        <h3 className={styles.h3}>Mistakes it is built to catch</h3>
        <ul className={styles.modes}>
          {task.modes.map((m) => (
            <li key={m}>
              <code>{m}</code>
              <span>{task.mode_text[m]}</span>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  );
}
