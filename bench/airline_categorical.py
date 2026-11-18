"""GBM-perf airline (1M train, 100k test) with categorical columns: moonboost
(categorical splits, loss-guided growth) against LightGBM and XGBoost at the
same number of leaves. 100 rounds, eta 0.1, max_cat_to_onehot 32, one thread.
Usage: python bench/airline_categorical.py 255 1023 4095
moonboost's time is its own report (binning included, CSV parsing not);
LightGBM's and XGBoost's include building their Dataset / DMatrix."""
import numpy as np, csv, time, subprocess, sys, os
import xgboost as xgb, lightgbm as lgb
from sklearn.metrics import roc_auc_score
D=os.environ.get('GBMPERF', 'gbmperf')  # train-1m.csv and test.csv from s3://benchm-ml--main
B=os.environ.get('MOONBOOST', '_build/native/release/build/cmd/main/main.exe')
def load(p):
  with open(p) as f:
    r=csv.reader(f); h=next(r); return h,list(r)
h,tr=load(f'{D}/train-1m.csv'); _,te=load(f'{D}/test.csv')
cols={c:i for i,c in enumerate(h)}
maps={c:{v:i for i,v in enumerate(sorted({row[cols[c]] for row in tr+te}))} for c in ['UniqueCarrier','Origin','Dest']}
def enc(rows):
  X=np.empty((len(rows),10),np.float32)
  for j,c in enumerate(['Month','DayofMonth','DayOfWeek']): X[:,j]=[float(r[cols[c]][2:]) for r in rows]
  for j,c in enumerate(['DepTime','Distance']): X[:,3+j]=[float(r[cols[c]]) for r in rows]
  for j,c in enumerate(['UniqueCarrier','Origin','Dest']): X[:,5+j]=[maps[c][r[cols[c]]] for r in rows]
  X[:,8]=X[:,3]//100; X[:,9]=X[:,3]%100  # departure hour and minute
  return X
Xtr=enc(tr); Xte=enc(te); CAT=[0,1,2,5,6,7]
ytr=np.array([r[cols['dep_delayed_15min']]=='Y' for r in tr],np.float32)
yte=np.array([r[cols['dep_delayed_15min']]=='Y' for r in te])
if not os.path.exists(f'{D}/Xc.csv'):
  np.savetxt(f'{D}/Xc.csv',Xtr,delimiter=',',fmt='%.9g'); np.savetxt(f'{D}/Xc_test.csv',Xte,delimiter=',',fmt='%.9g')
  np.savetxt(f'{D}/y.csv',ytr,fmt='%g')
os.sched_setaffinity(0,{3})
for L in [int(a) for a in sys.argv[1:]] or [1023]:
  o=subprocess.run(['taskset','-c','3',B,'train',f'{D}/Xc.csv',f'{D}/y.csv','-o','/tmp/ac_mb.json','--objective','binary:logistic','--rounds','100','--eta','0.1',
    '--grow-policy','lossguide','--depth','0','--max-leaves',str(L),'--categorical','0,1,2,5,6,7','--max-cat-to-onehot','32'],capture_output=True,text=True,check=True)
  ms=float(o.stdout.split(' in ')[1].split()[0])
  p=np.array(subprocess.run([B,'predict','/tmp/ac_mb.json',f'{D}/Xc_test.csv'],capture_output=True,text=True).stdout.split(),dtype=float)
  print('%-10s %5d leaves %6.1f s  AUC %.4f'%('moonboost',L,ms/1000,roc_auc_score(yte,p)),flush=True)
  t=time.perf_counter(); m=lgb.train({'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':L,'max_cat_to_onehot':32},
    lgb.Dataset(Xtr,label=ytr,categorical_feature=CAT),100)
  print('%-10s %5d leaves %6.1f s  AUC %.4f'%('lightgbm',L,time.perf_counter()-t,roc_auc_score(yte,m.predict(Xte))),flush=True)
  ft=['c' if j in CAT else 'q' for j in range(10)]
  t=time.perf_counter(); m=xgb.train({'objective':'binary:logistic','eta':0.1,'tree_method':'hist','nthread':1,'grow_policy':'lossguide','max_depth':0,
    'max_leaves':L,'max_cat_to_onehot':32},xgb.DMatrix(Xtr,label=ytr,feature_types=ft,enable_categorical=True,nthread=1),100)
  el=time.perf_counter()-t
  print('%-10s %5d leaves %6.1f s  AUC %.4f'%('xgboost',L,el,roc_auc_score(yte,m.predict(xgb.DMatrix(Xte,feature_types=ft,enable_categorical=True)))),flush=True)
