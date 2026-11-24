"""Peak memory (max RSS) of training on the GBM-perf airline data, one
process per run, one thread: moonboost's CLI against LightGBM and XGBoost
from Python, with the same settings as bench/airline_categorical.py (and
depth 10 on the label-encoded columns for GBM-perf). The Python processes
also hold the interpreter, numpy, pandas, the library and the float32 data;
the line "python baseline" is such a process with everything imported and
the data loaded, before any Dataset or DMatrix. moonboost's CLI holds its
float32 data matrix too.
Usage: GBMPERF=... MOONBOOST=... python bench/peak_memory.py 255 1023 4095 gbm"""
import os, sys, subprocess
D = os.environ.get('GBMPERF', 'gbmperf')
B = os.environ.get('MOONBOOST', '_build/native/release/build/cmd/main/main.exe')
CHILD = r'''
import sys, resource, numpy as np, pandas as pd
lib, L, D = sys.argv[1], int(sys.argv[2]), sys.argv[3]
X = pd.read_csv(D + ('/Xc.csv' if L else '/X.csv'), header=None).to_numpy(np.float32)
y = pd.read_csv(D + '/y.csv', header=None).to_numpy(np.float32).ravel()
CAT = [0, 1, 2, 5, 6, 7]
import lightgbm as lgb, xgboost as xgb
if lib == 'lgb':
    p = {'objective': 'binary', 'learning_rate': 0.1, 'num_threads': 1, 'verbose': -1}
    if L:
        p.update(num_leaves=L, max_cat_to_onehot=32)
        lgb.train(p, lgb.Dataset(X, label=y, categorical_feature=CAT), 100)
    else:
        p.update(num_leaves=1024, max_depth=10)
        lgb.train(p, lgb.Dataset(X, label=y), 100)
elif lib == 'xgb':
    p = {'objective': 'binary:logistic', 'eta': 0.1, 'tree_method': 'hist', 'nthread': 1}
    if L:
        p.update(grow_policy='lossguide', max_depth=0, max_leaves=L, max_cat_to_onehot=32)
        d = xgb.DMatrix(X, label=y, feature_types=['c' if j in CAT else 'q' for j in range(10)],
                        enable_categorical=True, nthread=1)
    else:
        p.update(max_depth=10)
        d = xgb.DMatrix(X, label=y, nthread=1)
    xgb.train(p, d, 100)
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
'''

def peak(args):
    p = subprocess.Popen(['taskset', '-c', '3'] + args, stdout=subprocess.PIPE, text=True)
    out = p.stdout.read()
    _, _, ru = os.wait4(p.pid, 0)
    return ru.ru_maxrss // 1024, out

for a in sys.argv[1:] or ['1023']:
    L = 0 if a == 'gbm' else int(a)
    name = 'GBM-perf depth 10' if L == 0 else f'{L} leaves, categories'
    mb = [B, 'train', f'{D}/' + ('Xc.csv' if L else 'X.csv'), f'{D}/y.csv', '-o', '/tmp/pm.json',
          '--objective', 'binary:logistic', '--rounds', '100', '--eta', '0.1']
    mb += (['--grow-policy', 'lossguide', '--depth', '0', '--max-leaves', str(L),
            '--categorical', '0,1,2,5,6,7', '--max-cat-to-onehot', '32'] if L else ['--depth', '10'])
    print(f'{name:24s} moonboost       {peak(mb)[0]:5d} MB', flush=True)
    for lib in ['base', 'lgb', 'xgb']:
        m, out = peak([sys.executable, '-c', CHILD, lib, str(L), D])
        label = {'base': 'python baseline', 'lgb': 'LightGBM', 'xgb': 'XGBoost'}[lib]
        print(f'{name:24s} {label:15s} {int(out.split()[-1]):5d} MB', flush=True)
