import { useEffect, useRef, useState } from "react";

import { useTrace } from "../../shared/api/queries";
import { us } from "../../shared/lib/format";
import { QueryView } from "../../shared/ui/states";
import styles from "./TraceView.module.css";

const LANE_H = 22;
const MIN_W = 0.5;

/** The task's trace as the solver sees it in a trace viewer: one lane per thread. */
export function TraceView({ taskId }: { taskId: string }) {
  const q = useTrace(taskId, true);
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, e?.contentRect.width ?? 800)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={box}>
      <QueryView query={q} what="the trace" lines={6} isEmpty={(d) => d.lanes.length === 0}>
        {(d) => {
          const span = Math.max(d.end_us - d.start_us, 1);
          const scale = width / span;
          return (
            <div className={styles.wrap}>
              <p className={styles.caption}>
                {d.lanes.length} lanes over {us(span)}. Hover an event for its name and duration.
              </p>
              <div className={styles.lanes}>
                {d.lanes.map((lane) => (
                  <div key={lane.name} className={styles.lane}>
                    <span className={styles.label} title={lane.name}>
                      {lane.name}
                    </span>
                    <svg width={width} height={LANE_H} role="img" aria-label={`${lane.name}: ${lane.events.length} events`} className={styles.svg}>
                      {lane.events.map((e, i) => (
                        <rect key={i} x={e.ts * scale} y={3} width={Math.max(MIN_W, e.dur * scale)} height={LANE_H - 6} className={styles.event}>
                          <title>{`${e.name}: ${us(e.dur)}`}</title>
                        </rect>
                      ))}
                    </svg>
                  </div>
                ))}
              </div>
              <table className={styles.table}>
                <caption>Longest totals</caption>
                <thead>
                  <tr>
                    <th scope="col">Event</th>
                    <th scope="col">Thread</th>
                    <th scope="col">Count</th>
                    <th scope="col">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {d.summary.map((r) => (
                    <tr key={`${r.thread}/${r.name}`}>
                      <td>
                        <code>{r.name}</code>
                      </td>
                      <td>{r.thread}</td>
                      <td>{r.count}</td>
                      <td>{us(r.total_us)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }}
      </QueryView>
    </div>
  );
}
