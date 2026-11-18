# run moonboost CLI on airline (+hour/minute), score test AUC with moonboost's own JSON via XGBoost and via moonboost predict
import numpy as np, subprocess, sys, os, time, json
from sklearn.metrics import roc_auc_score
os.chdir(os.path.expanduser('~/mb/acc'))
B=os.path.expanduser('~/mb/moonboost/_build/native/release/build/cmd/main/main.exe')
if not os.path.exists('Xh.csv'):
  Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy')
  hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]
  np.savetxt('Xh.csv',hr(Xtr),delimiter=',',fmt='%.9g'); np.savetxt('Xh_te.csv',hr(Xte),delimiter=',',fmt='%.9g')
  np.savetxt('y.csv',np.load('/tmp/ytr.npy'),fmt='%g')
yte=np.load('/tmp/yte.npy')
name=sys.argv[1]; args=sys.argv[2:]; core=os.environ.get('CORE','3')
t=time.perf_counter()
o=subprocess.run(['taskset','-c',core,B,'train','Xh.csv','y.csv','-o',f'm_{name}.json','--objective','binary:logistic','--rounds','100','--eta','0.1']+args,capture_output=True,text=True)
el=time.perf_counter()-t
if o.returncode: print(name,'FAIL',o.stdout[-500:],o.stderr[-800:]); sys.exit(1)
ms=o.stdout.split(' in ')[1].split()[0] if ' in ' in o.stdout else '?'
p=subprocess.run([B,'predict',f'm_{name}.json','Xh_te.csv'],capture_output=True,text=True).stdout.split()
auc=roc_auc_score(yte,np.array(p,dtype=float))
x='?'
try:
  import xgboost as xgb
  b=xgb.Booster(model_file=f'm_{name}.json'); px=b.predict(xgb.DMatrix(np.loadtxt('Xh_te.csv',delimiter=',',dtype=np.float32)))
  x='%.4f same=%s'%(roc_auc_score(yte,px),np.array_equal(px.astype(np.float32),np.array(p,dtype=np.float32)))
except Exception as e: x='xgb err '+str(e)[:150]
print('%-28s train %s ms  AUC %.4f  xgb %s  %s'%(name,ms,auc,x,' '.join(args)),flush=True)
