# idea R1: after boosting, backfit every tree's leaf values jointly (cyclic Newton per tree)
import numpy as np, time, os, sys, lightgbm as lgb
from sklearn.metrics import roc_auc_score
Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy'); ytr=np.load('/tmp/ytr.npy').astype(np.float64); yte=np.load('/tmp/yte.npy')
os.sched_setaffinity(0,{1})
hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]
CAT=[0,1,2,5,6,7]
A,B=hr(Xtr),hr(Xte)
p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':512,'cat_smooth':1}
m=lgb.train(p,lgb.Dataset(A,label=ytr,categorical_feature=CAT),100)
print('base AUC %.4f'%roc_auc_score(yte,m.predict(B)),flush=True)
Ltr=m.predict(A,pred_leaf=True).astype(np.int64); Lte=m.predict(B,pred_leaf=True).astype(np.int64)
T=Ltr.shape[1]
# per-tree leaf values from the model dump
d=m.dump_model()['tree_info']
def leafvals(tr):
  out={}
  def w(n):
    if 'leaf_index' in n: out[n['leaf_index']]=n['leaf_value']
    else: w(n['left_child']); w(n['right_child'])
  w(tr['tree_structure']); v=np.zeros(max(out)+1)
  for k,x in out.items(): v[k]=x
  return v
V=[leafvals(t) for t in d]
init=0.0
def margins(L,V): return sum(V[t][L[:,t]] for t in range(T))
mtr=margins(Ltr,V); mte=margins(Lte,V)
print('check AUC %.4f'%roc_auc_score(yte,mte),flush=True)
lam=float(sys.argv[1]) if len(sys.argv)>1 else 10.0; shrink=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
for ps in range(6):
  for t in range(T):
    mtr-=V[t][Ltr[:,t]]; mte-=V[t][Lte[:,t]]
    pr=1/(1+np.exp(-mtr)); g=pr-ytr; h=pr*(1-pr)
    k=len(V[t]); G=np.bincount(Ltr[:,t],g,k); H=np.bincount(Ltr[:,t],h,k)
    # one Newton step from the current values toward the refit
    pc=1/(1+np.exp(-(mtr+V[t][Ltr[:,t]]))); g2=pc-ytr; h2=pc*(1-pc)
    G2=np.bincount(Ltr[:,t],g2,k); H2=np.bincount(Ltr[:,t],h2,k)
    V[t]=V[t]-shrink*(G2+lam*V[t])/(H2+lam)
    mtr+=V[t][Ltr[:,t]]; mte+=V[t][Lte[:,t]]
  print('lam %g shrink %g pass %d train-AUC %.4f AUC %.4f'%(lam,shrink,ps+1,roc_auc_score(ytr,mtr),roc_auc_score(yte,mte)),flush=True)
