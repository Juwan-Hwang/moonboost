// moonboost's JS build against onnxruntime-web in headless Chromium, the
// browser counterpart of web_bench.mjs. Each engine and model gets a fresh
// browser context, so cold start includes import, model fetch and load:
//
//   python3 bench/gen_models.py && python3 bench/to_onnx.py
//   moon build --target js --release
//   (cd bench && npm install && npx playwright install chromium)
//   node bench/browser_bench.mjs [runs]
//
// Files are served from the repository root by a small static server on
// localhost. Timer resolution in the page is the browser's (cross-origin
// isolation is enabled so performance.now() is not coarsened to 100 us).
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const types = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json", ".csv": "text/csv",
};
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  try {
    const body = await readFile(join(root, path));
    res.writeHead(200, {
      "content-type": types[extname(path)] ?? "application/octet-stream",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}/bench/browser_bench.html`;

const runs = Number(process.argv[2] ?? 2);
const browser = await chromium.launch();
console.log(`# ${browser.version()}`);
for (const model of ["small", "medium", "large"]) {
  for (const engine of ["moonboost", "ort"]) {
    for (let k = 0; k < runs; k++) {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      page.on("pageerror", (e) => console.error(e));
      await page.goto(`${base}?engine=${engine}&model=${model}`);
      const r = await page.waitForFunction(() => window.__result, null, { timeout: 600000 });
      console.log(JSON.stringify(await r.jsonValue()));
      await ctx.close();
    }
  }
}
await browser.close();
server.close();
