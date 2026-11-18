# one moonboost config on one suite dataset: valid and test AUC (same split as suite_run)
import numpy as np, subprocess, os, sys
from sklearn.metrics import roc_auc_score
core=os.environ.get('CORE','0'); B=os.path.expanduser('~/mb/moonboost/_build/native/release/build/cmd/main/main.exe')
n=sys.argv[1]; args=sys.argv[2:]; W=os.path.expanduser(f'~/mb/acc/dse_{n}_{core}'); os.makedirs(W,exist_ok=True)
d=np.load(os.path.expanduser(f'~/mb/acc/ds/{n}.npz')); Xtr,Xte,ytr,yte,cats=d['Xtr'],d['Xte'],d['ytr'],d['yte'],d['cat']
rng=np.random.RandomState(1); idx=rng.permutation(len(ytr)); k=int(len(ytr)*0.8); fi,vi=idx[:k],idx[k:]
if not os.path.exists(f'{W}/X.csv'):
  np.savetxt(f'{W}/X.csv',Xtr[fi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',ytr[fi],fmt='%g')
  np.savetxt(f'{W}/v.csv',Xtr[vi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',Xte,delimiter=',',fmt='%.9g')
extra=['--categorical',','.join(map(str,cats))] if len(cats) else []
o=subprocess.run(['taskset','-c',core,B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective','binary:logistic','--rounds','100','--eta','0.1']+extra+args,capture_output=True,text=True)
if o.returncode: print(o.stdout[-300:],o.stderr[-300:]); sys.exit(1)
pr=lambda f: np.array(subprocess.run([B,'predict',f'{W}/m.json',f],capture_output=True,text=True).stdout.split(),dtype=float)
print('%-12s valid %.4f test %.4f  %s'%(n,roc_auc_score(ytr[vi],pr(f'{W}/v.csv')),roc_auc_score(yte,pr(f'{W}/t.csv')),' '.join(args)),flush=True)
