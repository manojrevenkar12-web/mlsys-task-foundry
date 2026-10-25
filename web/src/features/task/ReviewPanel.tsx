import { AnimatePresence, motion as m } from "framer-motion";
import { type FormEvent, useState } from "react";

import { ApiError } from "../../shared/api/client";
import { useAddReview, useReviews } from "../../shared/api/queries";
import type { Criterion, TaskDetail } from "../../shared/api/types";
import { motion } from "../../shared/theme/motion";
import { EmptyState, QueryView } from "../../shared/ui/states";
import styles from "./ReviewPanel.module.css";

type Verdict = "pass" | "fail" | undefined;

function CriterionRow({ c, value, onChange }: { c: Criterion; value: Verdict; onChange: (v: Verdict) => void }) {
  return (
    <fieldset className={styles.criterion}>
      <legend className={styles.text}>
        <span>{c.text}</span>
        <span className={styles.weight}>{c.weight} points</span>
      </legend>
      <div className={styles.anchors}>
        <p>
          <span className={styles.anchorLabel}>Passes:</span> {c.pass_anchor}
        </p>
        <p>
          <span className={styles.anchorLabel}>Fails:</span> {c.fail_anchor}
        </p>
      </div>
      <div className={styles.choice} role="radiogroup" aria-label={`Verdict: ${c.text}`}>
        {(["pass", "fail"] as const).map((v) => (
          <label key={v} className={styles.option} data-kind={v} data-checked={value === v || undefined}>
            <input type="radio" name={c.id} value={v} checked={value === v} onChange={() => onChange(v)} />
            {v === "pass" ? "Meets it" : "Does not"}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function ReviewPanel({ task }: { task: TaskDetail }) {
  const criteria = task.rubric?.criteria ?? [];
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [reviewer, setReviewer] = useState("");
  const [submission, setSubmission] = useState("reference");
  const [notes, setNotes] = useState("");
  const add = useAddReview(task.id);
  const list = useReviews(task.id);
  const complete = criteria.every((c) => verdicts[c.id]) && reviewer.trim() && submission.trim();

  if (!task.rubric) {
    return <EmptyState title="No written part to review">This task is graded entirely by its checks; see Proof.</EmptyState>;
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    if (!complete) return;
    add.mutate({
      task: task.id,
      submission: submission.trim(),
      reviewer: reviewer.trim(),
      verdicts: Object.fromEntries(criteria.map((c) => [c.id, verdicts[c.id] === "pass"])),
      notes,
    });
  }

  return (
    <div className={styles.grid}>
      <form className={styles.form} onSubmit={submit} aria-label="Review the written answer">
        <p className={styles.lede}>Judge the explanation in a submission's answer.yaml against each criterion. Two reviewers on the same submission give the rubric's agreement.</p>
        <div className={styles.who}>
          <label>
            Submission
            <input value={submission} onChange={(e) => setSubmission(e.target.value)} required maxLength={120} />
          </label>
          <label>
            Your name
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} required maxLength={80} autoComplete="name" />
          </label>
        </div>
        {criteria.map((c) => (
          <CriterionRow key={c.id} c={c} value={verdicts[c.id]} onChange={(v) => setVerdicts((s) => ({ ...s, [c.id]: v }))} />
        ))}
        <label className={styles.notes}>
          Notes for the author (optional)
          <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={3} maxLength={4000} />
        </label>
        <div className={styles.actions}>
          <button type="submit" className={styles.submit} disabled={!complete || add.isPending}>
            {add.isPending ? "Saving review…" : "Save review"}
          </button>
          {!complete && <span className={styles.hint}>Judge every criterion and fill in both names to save.</span>}
        </div>
        <AnimatePresence>
          {add.isSuccess && (
            <m.p key="ok" className={styles.saved} role="status" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: motion.base }}>
              Review saved: {add.data.score} of 100 points.
            </m.p>
          )}
          {add.isError && (
            <m.p key="err" className={styles.failed} role="alert" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              Could not save the review: {add.error instanceof ApiError ? add.error.message : "the API did not answer"}.
            </m.p>
          )}
        </AnimatePresence>
      </form>

      <section aria-labelledby="past-h" className={styles.past}>
        <h3 id="past-h" className={styles.h3}>
          Reviews so far
        </h3>
        <QueryView query={list} what="reviews" lines={3} isEmpty={(d) => d.reviews.length === 0} empty={<EmptyState title="No reviews yet">Save the first one with the form.</EmptyState>}>
          {(d) => (
            <>
              <ul className={styles.reviews}>
                {d.reviews.map((r) => {
                  const passed = criteria.filter((c) => r.verdicts[c.id]).length;
                  return (
                    <li key={`${r.submission}/${r.reviewer}`}>
                      <span className={styles.reviewWho}>
                        {r.reviewer} on <code>{r.submission}</code>
                      </span>
                      <span className={styles.reviewScore}>
                        {passed} of {criteria.length} criteria met
                      </span>
                    </li>
                  );
                })}
              </ul>
              {d.agreement.length > 0 && (
                <table className={styles.agreement}>
                  <caption>Agreement between reviewers (Cohen's kappa)</caption>
                  <tbody>
                    {d.agreement.map((a) => (
                      <tr key={a.submission}>
                        <td>
                          <code>{a.submission}</code>
                        </td>
                        <td>{a.reviewers.join(" and ")}</td>
                        <td data-low={a.kappa !== null && a.kappa < 0.6 ? true : undefined}>{a.kappa === null ? "–" : a.kappa.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </QueryView>
      </section>
    </div>
  );
}
