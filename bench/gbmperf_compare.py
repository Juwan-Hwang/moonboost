"""GBM-perf airline (1M train, 100k test), one thread, depth 10, eta 0.1,
100 rounds, 256 bins. Categoricals are label-encoded so all three tools see
the same dense numeric matrix."""
import numpy as np, csv, time, subprocess, sys, os
import xgboost as xgb, lightgbm as lgb
from sklearn.metrics import roc_auc_score
D=os.environ.get('GBMPERF', 'gbmperf')  # train-1m.csv and test.csv from s3://benchm-ml--main
B=os.environ.get('MOONBOOST', '_build/native/release/build/cmd/main/main.exe')
R=int(sys.argv[1]) if len(sys.argv)>1 else 100
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
if not os.path.exists(f'{D}/X.csv'):
  np.savetxt(f'{D}/X.csv',Xtr,delimiter=',',fmt='%.9g'); np.savetxt(f'{D}/y.csv',ytr,fmt='%g')
res=[]
out=subprocess.run(['taskset','-c','3',B,'train',f'{D}/X.csv',f'{D}/y.csv','-o','/tmp/gp_mb.json','--objective','binary:logistic','--rounds',str(R),'--depth','10','--eta','0.1'],capture_output=True,text=True)
print(out.stdout.strip(), out.stderr[-300:], flush=True)
ms=float(out.stdout.split(' in ')[1].split()[0])
mb=xgb.Booster(model_file='/tmp/gp_mb.json')
res.append(('moonboost',ms/1000,roc_auc_score(yte,mb.predict(xgb.DMatrix(Xte)))))
print(res[-1], flush=True)
os.sched_setaffinity(0,{3})
t=time.perf_counter(); d=xgb.DMatrix(Xtr,label=ytr,nthread=1)
m=xgb.train({'objective':'binary:logistic','tree_method':'hist','max_depth':10,'eta':0.1,'max_bin':256,'nthread':1},d,R)
res.append(('xgboost hist',time.perf_counter()-t,roc_auc_score(yte,m.predict(xgb.DMatrix(Xte)))))
print(res[-1], flush=True)
t=time.perf_counter(); d=lgb.Dataset(Xtr,label=ytr,params={'max_bin':255,'verbose':-1})
m=lgb.train({'objective':'binary','num_leaves':1024,'max_depth':10,'learning_rate':0.1,'min_data_in_leaf':1,'min_sum_hessian_in_leaf':1,'lambda_l2':1,'num_threads':1,'verbose':-1,'force_row_wise':True},d,R)
res.append(('lightgbm depth10',time.perf_counter()-t,roc_auc_score(yte,m.predict(Xte))))
for r in res: print('%-20s %7.2f s  AUC %.4f'%r)
