// Runs in the page opened by browser_bench.mjs. Same measurements as
// web_bench.mjs: cold start (import + fetch model + load + first prediction),
// batch latency over 10,000 rows (best of five), one-row-per-call latency,
// and bit-exact agreement with XGBoost's probabilities.
const q = new URLSearchParams(location.search);
const engine = q.get("engine") ?? "moonboost";
const name = q.get("model") ?? "small";
const text = async (p) => (await fetch(p)).text();

const rows = (await text("data/input.csv")).trim().split("\n")
  .map((l) => l.split(",").map(Number));
const cols = rows[0].length;
const flat = new Float32Array(rows.flat());

async function moonboost() {
  const mb = await import("../_build/js/release/build/web/web.js");
  const m = mb.load_model(await text(`data/${name}.json`));
  return {
    batch: (x) => mb.predict_batch(m, x, cols),
    one: (row) => mb.predict(m, row)[0],
  };
}

async function onnxruntime() {
  const ort = await import("./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL("./node_modules/onnxruntime-web/dist/", location.href).href;
  const bytes = new Uint8Array(await (await fetch(`data/${name}.onnx`)).arrayBuffer());
  const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const run = async (x, n) => {
    const out = await s.run({ input: new ort.Tensor("float32", x, [n, cols]) });
    const p = out.probabilities.data;
    const r = new Float32Array(n);
    for (let i = 0; i < n; i++) r[i] = p[2 * i + 1];
    return r;
  };
  return {
    batch: (x) => run(x, x.length / cols),
    one: async (row) => (await run(Float32Array.from(row), 1))[0],
  };
}

const tImport = performance.now();
const model = engine === "ort" ? await onnxruntime() : await moonboost();
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
const t = performance.now();
for (let i = 0; i < n; i++) await model.one(rows[i]);
const single = ((performance.now() - t) * 1000) / n;

const expected = (await text(`data/${name}.expected.csv`)).trim().split("\n").map(Number);
let exact = 0;
let maxDiff = 0;
for (let i = 0; i < expected.length; i++) {
  if (Math.fround(expected[i]) === out[i]) exact++;
  maxDiff = Math.max(maxDiff, Math.abs(Math.fround(expected[i]) - out[i]));
}
window.__result = {
  engine, model: name, cold_ms: +cold.toFixed(1),
  batch_us: +((best * 1000) / rows.length).toFixed(2),
  single_us: +single.toFixed(2), exact, max_diff: maxDiff, rows: expected.length,
};
