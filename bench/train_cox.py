# survival:cox on synthetic right-censored data: moonboost CLI vs XGBoost, one thread
# MOONBOOST=path/to/moonboost python bench/train_cox.py
import numpy as np, subprocess, time, xgboost as xgb, re, os
B = os.environ['MOONBOOST']
rng = np.random.default_rng(0)
n, d = 250000, 10
X = rng.normal(size=(n, d)).astype(np.float32)
risk = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.6 * np.sin(2 * X[:, 2]) + 0.4 * X[:, 3] * X[:, 4]
t = rng.weibull(1.5, n) * np.exp(-risk / 1.5)
c = rng.exponential(np.median(t) * 2.5, n)
y = np.where(t <= c, t, -c).astype(np.float32)
ntr = 200000
Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
np.savetxt('/tmp/c_X.csv', Xtr, delimiter=',', fmt='%.9g')
np.savetxt('/tmp/c_y.csv', ytr, fmt='%.9g')
np.savetxt('/tmp/c_Xt.csv', Xte[:10000], delimiter=',', fmt='%.9g')

def cindex(pred, y, m=6000):
    # Harrell's C on the first m test rows: higher hazard, earlier event
    p, y = pred[:m], y[:m]
    tt, ev = np.abs(y), y > 0
    num = den = 0.0
    for i in np.nonzero(ev)[0]:
        later = tt > tt[i]
        den += later.sum()
        num += (p[i] > p[later]).sum() + 0.5 * (p[i] == p[later]).sum()
    return num / den

print('censored', f'{(y < 0).mean():.2f}')
for depth in (4, 6):
    R = 200
    r = subprocess.run([B, 'train', '/tmp/c_X.csv', '/tmp/c_y.csv', '-o', '/tmp/c_m.json', '--objective', 'survival:cox',
                        '--rounds', str(R), '--depth', str(depth), '--eta', '0.05'], capture_output=True, text=True)
    tm = int(re.search(r'in (\d+) ms', r.stdout + r.stderr)[1]) / 1000
    b = xgb.Booster(model_file='/tmp/c_m.json')
    pm = b.predict(xgb.DMatrix(Xte))
    mm = b.predict(xgb.DMatrix(Xte[:10000]), output_margin=True)
    q = np.array(subprocess.run([B, 'predict', '/tmp/c_m.json', '/tmp/c_Xt.csv', '--margin'], capture_output=True,
                                text=True).stdout.split(), dtype=np.float32)
    bit = bool((mm.view(np.uint32) == q.view(np.uint32)).all())
    t0 = time.time()
    bx = xgb.train({'objective': 'survival:cox', 'eta': 0.05, 'max_depth': depth, 'tree_method': 'hist', 'nthread': 1},
                   xgb.DMatrix(Xtr, ytr, nthread=1), R)
    tx = time.time() - t0
    px = bx.predict(xgb.DMatrix(Xte))
    print(f'depth {depth}: moonboost {tm:.1f} s C {cindex(pm, yte):.4f} bit-identical {bit} | '
          f'XGBoost {tx:.1f} s C {cindex(px, yte):.4f} | true risk C {cindex(np.exp(risk[ntr:]), yte):.4f}', flush=True)
