// Screenshots of the console for the README: node web/scripts-shots.cjs http://127.0.0.1:8000 docs/img
const { chromium } = require("playwright");

const base = process.argv[2] ?? "http://127.0.0.1:8000";
const out = process.argv[3] ?? "docs/img";
const shots = [
  ["overview", "/", 1440, 900],
  ["proof", "/tasks/paged-attention-decode?tab=proof", 1440, 900],
  ["task", "/tasks/flash-attention-causal", 1440, 900],
  ["trace", "/tasks/multihost-straggler?tab=trace", 1440, 900],
  ["review", "/tasks/host-bound-input?tab=review", 1440, 1000],
  ["results", "/results", 1440, 900],
  ["mobile-proof", "/tasks/dp-replicas-drift?tab=proof", 390, 844],
];

(async () => {
  const browser = await chromium.launch();
  for (const [name, path, w, h] of shots) {
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.goto(base + path, { waitUntil: "networkidle" });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${out}/${name}.png` });
    await page.close();
    console.log("wrote", name);
  }
  await browser.close();
})();
