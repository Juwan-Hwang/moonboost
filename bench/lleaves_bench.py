"""lleaves (LLVM-compiled LightGBM) on the models from lgb_models.py, one
thread: batch and single-row latency and how many predictions match
LightGBM bit for bit. Compiling the large model can take tens of minutes.

Run from bench/:  python lleaves_bench.py small medium [large]
"""
import sys
import time
import numpy as np
import lleaves

X = np.load("data/lgb_input.npy").astype(np.float64)
for name in sys.argv[1:]:
    ref = np.loadtxt(f"data/lgb_{name}.expected.csv")
    model = lleaves.Model(model_file=f"data/lgb_{name}.txt")
    t0 = time.time()
    model.compile(cache=f"data/lleaves_{name}.o")
    compile_s = time.time() - t0
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        pred = model.predict(X, n_jobs=1)
        best = min(best, time.perf_counter() - t0)
    single = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(2000):
            model.predict(X[i:i + 1], n_jobs=1)
        single = min(single, (time.perf_counter() - t0) / 2000)
    print(f"{name}: compile {compile_s:.1f} s, batch {best / len(X) * 1e6:.3f} us/row, "
          f"single {single * 1e6:.1f} us, exact {(pred == ref).sum()}/{len(X)}", flush=True)
