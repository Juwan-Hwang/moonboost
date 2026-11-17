// node bench/train_wasm_bench.mjs X.csv y.csv rounds out.json
// Times training in the WebAssembly GC module against the JS build on the
// same parsed data (CSV parsing is not timed).
import { readFileSync, writeFileSync } from "node:fs";
import { loadTrainer } from "../web/train_wasm.mjs";
import * as js from "../_build/js/release/build/web/web.js";
const [xf, yf, rounds = "30", out = "/tmp/wasm_train.json"] = process.argv.slice(2);
const lines = readFileSync(xf, "utf8").trim().split("\n");
const cols = lines[0].split(",").length;
const data = new Float64Array(lines.length * cols);
lines.forEach((l, r) => l.split(",").forEach((v, c) => { data[r * cols + c] = v === "" ? NaN : +v; }));
const labels = Float64Array.from(readFileSync(yf, "utf8").trim().split("\n"), Number);
const train = await loadTrainer(readFileSync(process.env.WASM ??
  new URL("../_build/wasm-gc/release/build/webgc/webgc.wasm", import.meta.url)));
let t = performance.now();
const a = train(data, cols, labels, "binary:logistic", -1, +rounds, 10, 0.1);
console.log(`wasm-gc ${Math.round(performance.now() - t)} ms`);
writeFileSync(out, a);
if (process.env.JS) {
  t = performance.now();
  const b = js.train(data, cols, labels, "binary:logistic", -1, +rounds, 10, 0.1);
  console.log(`js ${Math.round(performance.now() - t)} ms`, a === b ? "SAME" : "DIFF");
}
