# airline (+hour, minute), equal tree size for all libraries: 100 rounds, lr 0.1, one thread
import numpy as np, time, os, sys, subprocess, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
core=os.environ.get('CORE','0'); os.sched_setaffinity(0,{int(core)})
Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy'); ytr=np.load('/tmp/ytr.npy'); yte=np.load('/tmp/yte.npy')
hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]; A,B=hr(Xtr),hr(Xte); CAT=[0,1,2,5,6,7]
lib=sys.argv[1]; L=int(sys.argv[2]); oh=int(sys.argv[3]) if len(sys.argv)>3 else 32
t=time.perf_counter()
if lib=='lgb':
  m=lgb.train({'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':L,'max_cat_to_onehot':oh},lgb.Dataset(A,label=ytr,categorical_feature=CAT),100); p=m.predict(B)
elif lib=='xgb':
  ft=['c' if j in CAT else 'q' for j in range(A.shape[1])]
  m=xgb.train({'objective':'binary:logistic','eta':0.1,'tree_method':'hist','nthread':1,'grow_policy':'lossguide','max_depth':0,'max_leaves':L,'max_cat_to_onehot':oh},xgb.DMatrix(A,label=ytr,feature_types=ft,enable_categorical=True,nthread=1),100)
  p=m.predict(xgb.DMatrix(B,feature_types=ft,enable_categorical=True))
else:
  MB=os.environ.get('MB_BIN',os.path.expanduser('~/mb/mbx/_build/native/release/build/cmd/main/main.exe'))
  subprocess.run(['taskset','-c',core,MB,'train','Xh.csv','y.csv','-o',f'air_{core}.json','--objective','binary:logistic','--rounds','100','--eta','0.1','--grow-policy','lossguide','--depth','0','--max-leaves',str(L),'--categorical','0,1,2,5,6,7','--max-cat-to-onehot',str(oh)],check=True,capture_output=True)
  p=np.array(subprocess.run([MB,'predict',f'air_{core}.json','Xh_te.csv'],capture_output=True,text=True).stdout.split(),dtype=float)
print('%-4s leaves %5d onehot %3d  %6.1fs  AUC %.4f'%(lib,L,oh,time.perf_counter()-t,roc_auc_score(yte,p)),flush=True)
