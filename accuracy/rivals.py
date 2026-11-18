# accuracy baselines on GBM-perf 1M, 100 rounds, depth 10, eta 0.1, 1 thread
import numpy as np, csv, time, sys, os
import xgboost as xgb, lightgbm as lgb
from sklearn.metrics import roc_auc_score
D=os.path.expanduser('~/mb/gbmperf')
def load(p):
  with open(p) as f:
    r=csv.reader(f); h=next(r); return h,list(r)
h,tr=load(f'{D}/train-1m.csv'); _,te=load(f'{D}/test.csv')
cols={c:i for i,c in enumerate(h)}
maps={c:{v:i for i,v in enumerate(sorted({row[cols[c]] for row in tr+te}))} for c in ['UniqueCarrier','Origin','Dest']}
def enc(rows):
  X=np.empty((len(rows),8),np.float32)
  for j,c in enumerate(['Month','DayofMonth','DayOfWeek']): X[:,j]=[float(r[cols[c]][2:]) for r in rows]
  for j,c in enumerate(['DepTime','Distance']): X[:,3+j]=[float(r[cols[c]]) for r in rows]
  for j,c in enumerate(['UniqueCarrier','Origin','Dest']): X[:,5+j]=[maps[c][r[cols[c]]] for r in rows]
  return X
Xtr=enc(tr); Xte=enc(te)
ytr=np.array([r[cols['dep_delayed_15min']]=='Y' for r in tr],np.float32)
yte=np.array([r[cols['dep_delayed_15min']]=='Y' for r in te])
np.save('/tmp/Xtr.npy',Xtr); np.save('/tmp/Xte.npy',Xte); np.save('/tmp/ytr.npy',ytr); np.save('/tmp/yte.npy',yte)
os.sched_setaffinity(0,{3}); R=100
which=sys.argv[1:] or ['xgb','xgbcat','lgb','lgbcat','lgbcat_leaf','cb']
CAT=[0,1,2,5,6,7]
def rep(n,t,p): print('%-28s %7.1f s  AUC %.4f'%(n,time.perf_counter()-t,roc_auc_score(yte,p)),flush=True)
for w in which:
  t=time.perf_counter()
  if w=='xgb':
    m=xgb.train({'objective':'binary:logistic','tree_method':'hist','max_depth':10,'eta':0.1,'nthread':1},xgb.DMatrix(Xtr,label=ytr,nthread=1),R); rep(w,t,m.predict(xgb.DMatrix(Xte)))
  if w=='xgbcat':
    import pandas as pd
    def df(X):
      d=pd.DataFrame(X,columns=[f'f{i}' for i in range(8)])
      for c in CAT: d[f'f{c}']=pd.Categorical(X[:,c].astype(int),categories=range(int(max(Xtr[:,c].max(),Xte[:,c].max()))+1))
      return d
    m=xgb.train({'objective':'binary:logistic','tree_method':'hist','max_depth':10,'eta':0.1,'nthread':1,'max_cat_to_onehot':1},xgb.DMatrix(df(Xtr),label=ytr,nthread=1,enable_categorical=True),R); rep(w,t,m.predict(xgb.DMatrix(df(Xte),enable_categorical=True)))
  if w.startswith('lgb'):
    cat=CAT if 'cat' in w else []
    p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1}
    if 'leaf' in w: p.update(num_leaves=512)
    else: p.update(num_leaves=1024,max_depth=10,min_data_in_leaf=1,min_sum_hessian_in_leaf=1,lambda_l2=1)
    m=lgb.train(p,lgb.Dataset(Xtr,label=ytr,categorical_feature=cat),R); rep(w,t,m.predict(Xte))
  if w=='cb':
    try:
      import catboost as cb
      m=cb.CatBoostClassifier(iterations=R,depth=10,learning_rate=0.1,thread_count=1,verbose=0)
      m.fit(Xtr.astype(int) if False else np.asarray(Xtr,dtype=object) if False else Xtr,ytr,cat_features=None)
      rep('cb nocat',t,m.predict_proba(Xte)[:,1])
      t=time.perf_counter(); import pandas as pd
      dtr=pd.DataFrame(Xtr); dte=pd.DataFrame(Xte)
      for c in CAT: dtr[c]=dtr[c].astype(int); dte[c]=dte[c].astype(int)
      m=cb.CatBoostClassifier(iterations=R,depth=10,learning_rate=0.1,thread_count=1,verbose=0); m.fit(dtr,ytr,cat_features=CAT); rep('cb cat',t,m.predict_proba(dte)[:,1])
    except ImportError: print('no catboost')
