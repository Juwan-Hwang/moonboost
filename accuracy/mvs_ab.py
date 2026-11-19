# MVS (gradient_based) vs uniform row sampling: per dataset and split, take the
# moonboost config picked in rep2 (SPACE=2), set subsample f and the method,
# 2 seeds each, report test AUC.
import numpy as np, os, sys, json, ast
sys.argv, ds = sys.argv[:1], sys.argv[1:]
import tuned as T
from sklearn.metrics import roc_auc_score
T.B=os.environ.get('MB_BIN',os.path.expanduser('~/mb/mbv/_build/native/release/build/cmd/main/main.exe'))
out=os.path.expanduser('~/mb/acc/mvs_ab.jsonl')
best={}
for l in open(os.path.expanduser('~/mb/acc/rep2.jsonl')):
  r=json.loads(l)
  if r['lib']=='moonboost': best[(r['ds'],r['split'])]=ast.literal_eval(r['cfg'])
for n in ds:
  d=np.load(os.path.expanduser(f'~/mb/acc/ds/{n}.npz'))
  X=np.vstack([d['Xtr'],d['Xte']]); y=np.concatenate([d['ytr'],d['yte']]); cats=d['cat']
  W=os.path.expanduser(f'~/mb/acc/mv_{n}_{T.core}'); os.makedirs(W,exist_ok=True)
  for s in range(3):
    r=np.random.RandomState(100+s); idx=r.permutation(len(y)); k=int(len(y)*0.8)
    tr,te=idx[:k],idx[k:]; k2=int(len(tr)*0.8); fi,vi=tr[:k2],tr[k2:]
    np.savetxt(f'{W}/X.csv',X[fi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',y[fi],fmt='%g')
    np.savetxt(f'{W}/v.csv',X[vi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[te],delimiter=',',fmt='%.9g')
    base=best[(n,s)]; i=base.index('--subsample'); base=base[:i]+base[i+2:]
    for f in (0.5,0.8):
      for m in ('uniform','gradient_based'):
        for seed in (0,1):
          cfg=base+['--subsample',str(f),'--sampling-method',m,'--seed',str(seed)]
          pv,pt=T.run('moonboost',cfg,X[fi],y[fi],[X[vi],X[te]],cats,W)
          rec={'ds':n,'split':s,'f':f,'m':m,'seed':seed,'valid':roc_auc_score(y[vi],pv),'test':roc_auc_score(y[te],pt)}
          print(json.dumps(rec),flush=True)
          with open(out,'a') as fo: fo.write(json.dumps(rec)+'\n')
