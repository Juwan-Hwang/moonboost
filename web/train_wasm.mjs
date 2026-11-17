// Training in a WebAssembly GC module (webgc/): the same models as `train`
// in the ES module, about twice as fast. Needs WebAssembly GC: Chrome 119,
// Firefox 120, Safari 18.2, Node 22.
//
//   import { loadTrainer } from "./train_wasm.mjs";
//   const train = await loadTrainer("webgc.wasm");
//   const json = train(data, cols, labels, "binary:logistic", -1, 100, 6, 0.3);

const OBJECTIVES = {
  "reg:squarederror": 0,
  "binary:logistic": 1,
  "multi:softprob": 2,
  "multi:softmax": 3,
};

// `source` is the module's URL, a fetch Response, or its bytes.
export async function loadTrainer(source) {
  let instance;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    ({ instance } = await WebAssembly.instantiate(source));
  } else {
    const response = source instanceof Response ? source : fetch(source);
    ({ instance } = await WebAssembly.instantiateStreaming(response));
  }
  const x = instance.exports;
  const read = (n, at) => {
    let s = "";
    const chunk = new Array(Math.min(n, 8192));
    for (let i = 0; i < n; i += chunk.length) {
      const m = Math.min(chunk.length, n - i);
      chunk.length = m;
      for (let j = 0; j < m; j++) chunk[j] = at(i + j);
      s += String.fromCharCode.apply(null, chunk);
    }
    return s;
  };
  // Arguments as for `train` in the ES module: `data` holds the rows one
  // after another (NaN for missing), -1 or "" keeps XGBoost's default.
  // Returns the model as XGBoost JSON text; throws on bad input.
  return function train(data, cols, labels, objective = "", numClass = -1,
    numRound = -1, maxDepth = -1, eta = -1) {
    const rows = labels.length;
    if (data.length !== rows * cols) {
      throw new Error(`${data.length} values for ${rows} rows of ${cols}`);
    }
    const code = objective === "" ? -1 : OBJECTIVES[objective];
    if (code === undefined) throw new Error(`unsupported: training objective ${objective}`);
    x.train_alloc(rows, cols);
    for (let i = 0; i < data.length; i++) x.train_set(i, data[i]);
    for (let i = 0; i < rows; i++) x.train_label(i, labels[i]);
    const n = x.train_run(code, numClass, numRound, maxDepth, eta);
    x.train_alloc(0, 0);
    if (n < 0) throw new Error(read(x.train_error_char(-1), x.train_error_char));
    return read(n, x.train_char);
  };
}
