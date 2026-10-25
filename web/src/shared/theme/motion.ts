/** Motion constants, matching the CSS tokens. One orchestrated moment: the proof matrix filling in,
 * cell by cell, when a task opens. Everything else answers what the reviewer did. */
export const motion = {
  fast: 0.12,
  base: 0.22,
  cellStagger: 0.012,
  ease: [0.22, 1, 0.36, 1] as const,
  spring: { type: "spring" as const, stiffness: 420, damping: 32 },
};
