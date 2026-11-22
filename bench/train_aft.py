# survival:aft on synthetic censored data (right, left and interval censored rows):
# moonboost CLI vs XGBoost, one thread. MOONBOOST=path/to/moonboost python bench/train_aft.py
import numpy as np, subprocess, time, xgboost as xgb, re, os
B = os.environ['MOONBOOST']
rng = np.random.default_rng(0)
n, d = 250000, 10
X = rng.normal(size=(n, d)).astype(np.float32)
mu = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.6 * np.sin(2 * X[:, 2]) + 0.4 * X[:, 3] * X[:, 4]
t = np.exp(mu + 0.5 * rng.logistic(size=n))
lo, hi = t.copy(), t.copy()
k = rng.random(n)
c = rng.exponential(np.median(t) * 2.5, n)
right = (k < 0.6) & (c < t)                   # right censored at c
lo[right], hi[right] = c[right], np.inf
left = (k >= 0.6) & (k < 0.7)                 # left censored: before a check
hi[left] = t[left] * (1 + rng.random(left.sum())); lo[left] = 0
inter = (k >= 0.7) & (k < 0.85)               # between two checks
w = rng.random(inter.sum())
lo[inter], hi[inter] = t[inter] * (1 - 0.5 * w), t[inter] * (1 + w)
lo, hi = lo.astype(np.float32), hi.astype(np.float32)
ntr = 200000
np.savetxt('/tmp/a_X.csv', X[:ntr], delimiter=',', fmt='%.9g')
np.savetxt('/tmp/a_lo.csv', lo[:ntr], fmt='%.9g')
np.savetxt('/tmp/a_hi.csv', hi[:ntr], fmt='%.9g')
np.savetxt('/tmp/a_Xt.csv', X[ntr:ntr + 10000], delimiter=',', fmt='%.9g')
dte = xgb.DMatrix(X[ntr:]); dte.set_float_info('label_lower_bound', lo[ntr:]); dte.set_float_info('label_upper_bound', hi[ntr:])
dtr = xgb.DMatrix(X[:ntr], nthread=1)
dtr.set_float_info('label_lower_bound', lo[:ntr]); dtr.set_float_info('label_upper_bound', hi[:ntr])
obs = lo[ntr:] == hi[ntr:]
def rmse_log(b):  # on the test rows with observed times
    return float(np.sqrt(np.mean((np.log(b.predict(dte))[obs] - np.log(t[ntr:][obs])) ** 2)))
print(f'observed {np.mean(lo == hi):.2f} right {right.mean():.2f} left {left.mean():.2f} interval {inter.mean():.2f}')
for dist, scale in (('normal', 1.0), ('logistic', 0.5), ('extreme', 0.8)):
    for depth in (4, 6):
        R = 200
        p = {'objective': 'survival:aft', 'aft_loss_distribution': dist, 'aft_loss_distribution_scale': scale,
             'eta': 0.05, 'max_depth': depth, 'tree_method': 'hist', 'nthread': 1}
        r = subprocess.run([B, 'train', '/tmp/a_X.csv', '/tmp/a_lo.csv', '-o', '/tmp/a_m.json', '--objective', 'survival:aft',
                            '--label-upper-bound', '/tmp/a_hi.csv', '--aft-distribution', dist, '--aft-scale', str(scale),
                            '--rounds', str(R), '--depth', str(depth), '--eta', '0.05'], capture_output=True, text=True)
        if r.returncode: print(r.stdout, r.stderr); raise SystemExit(1)
        tm = int(re.search(r'in (\d+) ms', r.stdout + r.stderr)[1]) / 1000
        b = xgb.Booster(model_file='/tmp/a_m.json'); b.set_param(p)
        mm = b.predict(xgb.DMatrix(X[ntr:ntr + 10000]))
        q = np.array(subprocess.run([B, 'predict', '/tmp/a_m.json', '/tmp/a_Xt.csv'], capture_output=True,
                                    text=True).stdout.split(), dtype=np.float32)
        bit = bool((mm.view(np.uint32) == q.view(np.uint32)).all())
        nm = float(b.eval(dte).split(':')[1])
        t0 = time.time(); bx = xgb.train(p, dtr, R); tx = time.time() - t0
        nx = float(bx.eval(dte).split(':')[1])
        print(f'{dist} {scale} depth {depth}: moonboost {tm:.1f} s nloglik {nm:.4f} rmse(log t) {rmse_log(b):.4f} bit-identical {bit} | '
              f'XGBoost {tx:.1f} s nloglik {nx:.4f} rmse {rmse_log(bx):.4f}', flush=True)
