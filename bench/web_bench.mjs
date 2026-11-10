// moonboost's JS build against onnxruntime-web (WebAssembly, one thread) in
// Node, on the benchmark models. Each engine runs in a fresh process so cold
// start is measured honestly:
//
//   python3 bench/gen_models.py && python3 bench/to_onnx.py
//   moon build --target js --release
//   (cd bench && npm install)
//   node bench/web_bench.mjs moonboost
//   node bench/web_bench.mjs ort
//
// Reported per model: cold start (import + load + first prediction), batch
// latency over 10,000 rows (best of five), one-row-per-call latency, and how
// many of the 10,000 probabilities equal XGBoost's bit for bit (and the
// largest absolute difference).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const t0 = performance.now();
const engine = process.argv[2] ?? "moonboost";
const here = dirname(fileURLToPath(import.meta.url));
const data = join(here, "data");
const rows = readFileSync(join(data, "input.csv"), "utf8").trim().split("\n")
  .map((l) => l.split(",").map(Number));
const cols = rows[0].length;
const flat = new Float32Array(rows.flat());
const setup = performance.now() - t0;

async function moonboost() {
  const mb = await import("../_build/js/release/build/web/web.js");
  return {
    async load(name) {
      const m = mb.load_model(readFileSync(join(data, `${name}.json`), "utf8"));
      return {
        batch: (x) => mb.predict_batch(m, x, cols),
        one: (row) => mb.predict(m, row)[0],
      };
    },
  };
}

async function onnxruntime() {
  const ort = await import("onnxruntime-web/wasm");
  ort.env.wasm.numThreads = 1;
  return {
    async load(name) {
      const s = await ort.InferenceSession.create(
        readFileSync(join(data, `${name}.onnx`)),
        { executionProviders: ["wasm"] },
      );
      const run = async (x, n) => {
        const out = await s.run({ input: new ort.Tensor("float32", x, [n, cols]) });
        const p = out.probabilities.data; // [n, 2]
        const r = new Float32Array(n);
        for (let i = 0; i < n; i++) r[i] = p[2 * i + 1];
        return r;
      };
      return {
        batch: (x) => run(x, x.length / cols),
        one: async (row) => (await run(Float32Array.from(row), 1))[0],
      };
    },
  };
}

const name = process.argv[3] ?? "small";
const tImport = performance.now();
const api = engine === "ort" ? await onnxruntime() : await moonboost();
const model = await api.load(name);
await model.one(rows[0]);
const cold = performance.now() - tImport;

const out = await model.batch(flat);
let best = Infinity;
for (let k = 0; k < 5; k++) {
  const t = performance.now();
  await model.batch(flat);
  best = Math.min(best, performance.now() - t);
}
const n = 2000;
for (let i = 0; i < n; i++) await model.one(rows[i]);
let t = performance.now();
for (let i = 0; i < n; i++) await model.one(rows[i]);
const single = ((performance.now() - t) * 1000) / n;

const expected = readFileSync(join(data, `${name}.expected.csv`), "utf8")
  .trim().split("\n").map(Number);
let exact = 0;
let maxDiff = 0;
for (let i = 0; i < expected.length; i++) {
  if (Math.fround(expected[i]) === out[i]) exact++;
  maxDiff = Math.max(maxDiff, Math.abs(Math.fround(expected[i]) - out[i]));
}
console.log(JSON.stringify({
  engine, model: name, cold_ms: +cold.toFixed(1),
  batch_us: +((best * 1000) / rows.length).toFixed(2),
  single_us: +single.toFixed(2), exact, max_diff: maxDiff, rows: expected.length, setup_ms: +setup.toFixed(0),
}));
