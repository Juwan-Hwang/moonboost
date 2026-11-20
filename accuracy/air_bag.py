# clean timing + AUC on airline (+hour, minute) at equal leaves: moonboost CLI train time (binning included,
# CSV parsing excluded) vs LightGBM / XGBoost (Dataset construction included). One thread, one core.
import numpy as np, time, os, sys, subprocess, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
core=os.environ.get('CORE','1'); os.sched_setaffinity(0,{int(core)}); os.chdir(os.path.expanduser('~/mb/acc'))
Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy'); ytr=np.load('/tmp/ytr.npy'); yte=np.load('/tmp/yte.npy')
hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]; A,B=hr(Xtr),hr(Xte); CAT=[0,1,2,5,6,7]
MB=os.environ.get('MB_BIN',os.path.expanduser('~/mb/mbx/_build/native/release/build/cmd/main/main.exe'))
for L in [int(x) for x in sys.argv[1:]]:
  o=subprocess.run(['taskset','-c',core,MB,'train','Xh.csv','y.csv','-o','at.json','--objective','binary:logistic','--rounds','100','--eta','0.1','--grow-policy','lossguide','--depth','0','--max-leaves',str(L),'--categorical','0,1,2,5,6,7','--max-cat-to-onehot','32','--subsample','0.5'],capture_output=True,text=True)
  ms=float(o.stdout.split(' in ')[1].split()[0])/1000
  p=np.array(subprocess.run([MB,'predict','at.json','Xh_te.csv'],capture_output=True,text=True).stdout.split(),dtype=float)
  print('moonboost leaves %5d  %6.1fs  AUC %.4f'%(L,ms,roc_auc_score(yte,p)),flush=True)
  t=time.perf_counter(); m=lgb.train({'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':L,'max_cat_to_onehot':32,'bagging_fraction':0.5,'bagging_freq':1},lgb.Dataset(A,label=ytr,categorical_feature=CAT),100)
  print('lightgbm  leaves %5d  %6.1fs  AUC %.4f'%(L,time.perf_counter()-t,roc_auc_score(yte,m.predict(B))),flush=True)
  ft=['c' if j in CAT else 'q' for j in range(A.shape[1])]
  t=time.perf_counter(); m=xgb.train({'objective':'binary:logistic','eta':0.1,'tree_method':'hist','nthread':1,'grow_policy':'lossguide','max_depth':0,'max_leaves':L,'max_cat_to_onehot':32,'subsample':0.5},xgb.DMatrix(A,label=ytr,feature_types=ft,enable_categorical=True,nthread=1),100)
  el=time.perf_counter()-t
  print('xgboost   leaves %5d  %6.1fs  AUC %.4f'%(L,el,roc_auc_score(yte,m.predict(xgb.DMatrix(B,feature_types=ft,enable_categorical=True)))),flush=True)
