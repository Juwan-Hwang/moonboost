# Multi-dataset accuracy benchmark: 100 rounds, lr 0.1, each library picks its best config on a
# validation split of the training data (same grid size), test AUC reported. One thread.
import numpy as np, subprocess, os, sys, time, json, glob
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
os.sched_setaffinity(0,{int(os.environ.get('CORE',0))}); core=os.environ.get('CORE','0')
B=os.path.expanduser('~/mb/moonboost/_build/native/release/build/cmd/main/main.exe')
W=os.path.expanduser('~/mb/acc/suite_tmp'+core); os.makedirs(W,exist_ok=True)
names=sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.expanduser('~/mb/acc/ds/*.npz')))
def mb_fit(Xf,yf,Xs,cats,args):
  np.savetxt(f'{W}/X.csv',Xf,delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',yf,fmt='%g')
  extra=['--categorical',','.join(map(str,cats))] if len(cats) else []
  o=subprocess.run(['taskset','-c',core,B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective','binary:logistic','--rounds','100','--eta','0.1']+extra+args,capture_output=True,text=True)
  if o.returncode: raise RuntimeError(o.stdout[-300:]+o.stderr[-300:])
  out=[]
  for X in Xs:
    np.savetxt(f'{W}/t.csv',X,delimiter=',',fmt='%.9g')
    out.append(np.array(subprocess.run([B,'predict',f'{W}/m.json',f'{W}/t.csv'],capture_output=True,text=True).stdout.split(),dtype=float))
  return out
def lgb_fit(Xf,yf,Xs,cats,kw):
  p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1}; p.update(kw)
  m=lgb.train(p,lgb.Dataset(Xf,label=yf,categorical_feature=list(cats)),100); return [m.predict(X) for X in Xs]
def xgb_fit(Xf,yf,Xs,cats,kw):
  ft=['c' if j in set(cats) else 'q' for j in range(Xf.shape[1])]
  p={'objective':'binary:logistic','eta':0.1,'tree_method':'hist','nthread':1}; p.update(kw)
  m=xgb.train(p,xgb.DMatrix(Xf,label=yf,feature_types=ft,enable_categorical=True,nthread=1),100)
  return [m.predict(xgb.DMatrix(X,feature_types=ft,enable_categorical=True)) for X in Xs]
GRID={
 'moonboost':[(mb_fit,a) for a in (['--depth','6'],['--grow-policy','lossguide','--depth','0','--max-leaves','31','--max-cat-to-onehot','32'],['--grow-policy','lossguide','--depth','0','--max-leaves','255','--max-cat-to-onehot','32'],['--grow-policy','lossguide','--depth','0','--max-leaves','1023','--max-cat-to-onehot','32'])],
 'lightgbm':[(lgb_fit,k) for k in ({'num_leaves':31},{'num_leaves':255,'max_cat_to_onehot':32},{'num_leaves':1023,'max_cat_to_onehot':32},{'num_leaves':255})],
 'xgboost':[(xgb_fit,k) for k in ({'max_depth':6},{'max_depth':10,'max_cat_to_onehot':32},{'grow_policy':'lossguide','max_depth':0,'max_leaves':255,'max_cat_to_onehot':32},{'grow_policy':'lossguide','max_depth':0,'max_leaves':1023,'max_cat_to_onehot':32})],
}
for n in names:
  d=np.load(os.path.expanduser(f'~/mb/acc/ds/{n}.npz')); Xtr,Xte,ytr,yte,cats=d['Xtr'],d['Xte'],d['ytr'],d['yte'],d['cat']
  rng=np.random.RandomState(1); idx=rng.permutation(len(ytr)); k=int(len(ytr)*0.8); fi,vi=idx[:k],idx[k:]
  # categorical codes only for columns with integer codes (as saved)
  res={}
  for lib,grid in GRID.items():
    best=None; t=time.perf_counter()
    for fn,cfg in grid:
      try:
        pv,pt=fn(Xtr[fi],ytr[fi],[Xtr[vi],Xte],cats,cfg); v=roc_auc_score(ytr[vi],pv); a=roc_auc_score(yte,pt)
      except Exception as e:
        print(n,lib,cfg,'ERR',str(e)[:200],flush=True); continue
      if best is None or v>best[0]: best=(v,a,cfg)
    res[lib]=best; print('%-14s %-10s valid %.4f test %.4f  %s  (%.0fs)'%(n,lib,best[0],best[1],best[2],time.perf_counter()-t),flush=True)
  with open(os.path.expanduser('~/mb/acc/suite_results.jsonl'),'a') as f: f.write(json.dumps({'ds':n,**{k:[v[0],v[1],str(v[2])] for k,v in res.items() if v}})+'\n')
