// Run after `moon build --target js --release`:
//   node examples/node/iris.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as mb from "../../_build/js/release/build/web/web.js";

const here = dirname(fileURLToPath(import.meta.url));
const model = mb.load_model(readFileSync(join(here, "../iris/model.json"), "utf8"));
if (model === undefined) throw new Error(mb.last_error());

const rows = [
  [5.1, 3.5, 1.4, 0.2],
  [6.3, 2.5, 4.9, 1.5],
  [5.9, NaN, 4.2, NaN],
];
for (const row of rows) {
  const p = mb.predict(model, row);
  console.log(row.join(","), "->", p.map((x) => x.toFixed(4)).join(","));
}
const batch = mb.predict_batch(model, rows.flat(), 4);
console.log("batch:", batch.length / mb.output_width(model), "rows");
