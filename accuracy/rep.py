# Repeated-split equal-budget tuning: merge train+test, draw SPLITS random 80/20 splits,
# for each split every library gets the same TRIALS random configs (picked on an inner
# validation split), report mean test AUC and the paired difference vs moonboost.
import numpy as np, os, sys, json, time
sys.argv, ds = sys.argv[:1], sys.argv[1:]
import tuned as T  # reuses configs and runners (its main loop runs over no datasets)
import os as _os
if _os.environ.get('SPACE')=='2': import space2  # adds subsample and colsample_bynode
from sklearn.metrics import roc_auc_score
T.B=os.environ.get('MB_BIN',os.path.expanduser('~/mb/mbx/_build/native/release/build/cmd/main/main.exe'))
SPLITS=int(os.environ.get('SPLITS','5')); TRIALS=int(os.environ.get('TRIALS','12'))
libs=os.environ.get('LIBS','moonboost,lightgbm,xgboost').split(',')
out=os.path.expanduser(os.environ.get('OUT','~/mb/acc/rep_results.jsonl'))
for n in ds:
  d=np.load(os.path.expanduser(f'~/mb/acc/ds/{n}.npz'))
  X=np.vstack([d['Xtr'],d['Xte']]); y=np.concatenate([d['ytr'],d['yte']]); cats=d['cat']
  W=os.path.expanduser(f'~/mb/acc/rw_{n}_{T.core}'); os.makedirs(W,exist_ok=True)
  for s in range(SPLITS):
    r=np.random.RandomState(100+s); idx=r.permutation(len(y)); k=int(len(y)*0.8)
    tr,te=idx[:k],idx[k:]; k2=int(len(tr)*0.8); fi,vi=tr[:k2],tr[k2:]
    np.savetxt(f'{W}/X.csv',X[fi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',y[fi],fmt='%g')
    np.savetxt(f'{W}/v.csv',X[vi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[te],delimiter=',',fmt='%.9g')
    for lib in libs:
      rr=np.random.RandomState(42+s); best=None; t=time.perf_counter()
      for i in range(TRIALS):
        cfg=T.CFG[lib](T.shared(rr))
        try: pv,pt=T.run(lib,cfg,X[fi],y[fi],[X[vi],X[te]],cats,W)
        except Exception: continue
        v=roc_auc_score(y[vi],pv)
        if best is None or v>best[0]: best=(v,roc_auc_score(y[te],pt),cfg)
      rec={'ds':n,'split':s,'lib':lib,'valid':best[0],'test':best[1],'cfg':str(best[2]),'trials':TRIALS,'sec':time.perf_counter()-t}
      print(json.dumps(rec),flush=True)
      with open(out,'a') as f: f.write(json.dumps(rec)+'\n')
