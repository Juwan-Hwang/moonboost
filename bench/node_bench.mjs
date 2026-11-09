// Time the JS build in Node on the benchmark models.
//   moon build --target js --release && node bench/node_bench.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as mb from "../_build/js/release/build/web/web.js";

const data = join(dirname(fileURLToPath(import.meta.url)), "data");
const rows = readFileSync(join(data, "input.csv"), "utf8").trim().split("\n")
  .map((l) => l.split(",").map(Number));
const cols = rows[0].length;
const flat = rows.flat();

for (const name of ["small", "medium", "large"]) {
  const text = readFileSync(join(data, `${name}.json`), "utf8");
  let t = performance.now();
  const model = mb.load_model(text);
  const load = performance.now() - t;
  t = performance.now();
  mb.predict(model, rows[0]);
  const first = performance.now() - t;

  // best of five timed batches after a warm-up, as the native and XGBoost
  // benchmarks report
  const out = mb.predict_batch(model, flat, cols);
  let best = Infinity;
  for (let k = 0; k < 5; k++) {
    t = performance.now();
    mb.predict_batch(model, flat, cols);
    best = Math.min(best, performance.now() - t);
  }
  const batch = (best * 1000) / rows.length;

  const n = 2000;
  for (let i = 0; i < n; i++) mb.predict(model, rows[i % rows.length]); // warm up
  t = performance.now();
  for (let i = 0; i < n; i++) mb.predict(model, rows[i % rows.length]);
  const single = ((performance.now() - t) * 1000) / n;

  const expected = readFileSync(join(data, `${name}.expected.csv`), "utf8")
    .trim().split("\n").map(Number);
  let exact = 0;
  for (let i = 0; i < expected.length; i++) {
    if (Math.fround(expected[i]) === out[i]) exact++;
  }
  console.log(
    `${name.padEnd(7)} load ${load.toFixed(0)} ms, first prediction ${first.toFixed(2)} ms, ` +
      `batch ${batch.toFixed(2)} us/row, single ${single.toFixed(2)} us/row, ` +
      `exact ${exact}/${expected.length}`,
  );
}
