# proxy experiments: which ideas raise AUC at 100 trees (LightGBM as the engine)
import numpy as np, time, sys, os, lightgbm as lgb
from sklearn.metrics import roc_auc_score
Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy'); ytr=np.load('/tmp/ytr.npy'); yte=np.load('/tmp/yte.npy')
os.sched_setaffinity(0,{int(os.environ.get("CORE",2))})
CAT=[0,1,2,5,6,7]
def run(name,A,B,cat,**kw):
  p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':512}; p.update(kw)
  t=time.perf_counter(); m=lgb.train(p,lgb.Dataset(A,label=ytr,categorical_feature=cat),100)
  print('%-36s %6.1fs AUC %.4f'%(name,time.perf_counter()-t,roc_auc_score(yte,m.predict(B))),flush=True)
def code(a,b,*cs):  # joint code of columns
  k=np.zeros(len(a),np.int64); k2=np.zeros(len(b),np.int64)
  for c in cs:
    m=int(max(a[:,c].max(),b[:,c].max()))+1; k=k*m+a[:,c].astype(np.int64); k2=k2*m+b[:,c].astype(np.int64)
  return k,k2
def te_oof(k,k2,folds=5,prior=20):
  g=ytr.mean(); out=np.empty(len(k)); rng=np.random.RandomState(0); f=rng.randint(0,folds,len(k))
  for i in range(folds):
    tr=f!=i; s=np.bincount(k[tr],ytr[tr],minlength=k.max()+1); c=np.bincount(k[tr],minlength=k.max()+1)
    out[~tr]=(s[k[~tr]]+prior*g)/(c[k[~tr]]+prior)
  s=np.bincount(k,ytr,minlength=max(k.max(),k2.max())+1); c=np.bincount(k,minlength=max(k.max(),k2.max())+1)
  return out,(s[k2]+prior*g)/(c[k2]+prior)
exp=sys.argv[1:] or ['base','hour','te_pairs']
hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]
for e in exp:
  if e=='base': run('lgb cat leaf512',Xtr,Xte,CAT)
  if e=='l1024': run('lgb cat leaf1024',Xtr,Xte,CAT,num_leaves=1024)
  if e=='hour': run('+hour,minute',hr(Xtr),hr(Xte),CAT)
  if e=='te_pairs':
    A=[Xtr];B=[Xte]
    for cs in [(5,6),(5,7),(6,7),(5,6,7),(0,1),(0,6),(2,5),(1,6)]:
      k,k2=code(Xtr,Xte,*cs); a,b=te_oof(k,k2); A.append(a[:,None]); B.append(b[:,None])
    run('+TE pairs',np.hstack(A),np.hstack(B),CAT)
  if e=='lr':
    for lr in [0.05,0.15,0.2]: run(f'lr {lr}',Xtr,Xte,CAT,learning_rate=lr)
  if e=='reg':
    run('cat_smooth 1',Xtr,Xte,CAT,cat_smooth=1); run('cat_l2 1',Xtr,Xte,CAT,cat_l2=1); run('min_data_per_group 10',Xtr,Xte,CAT,min_data_per_group=10,cat_smooth=5)
    run('min_data_in_leaf 100',Xtr,Xte,CAT,min_data_in_leaf=100)
  if e=='ceiling':
    for R2,lr in [(1000,0.05)]:
      p={'objective':'binary','learning_rate':lr,'num_threads':1,'verbose':-1,'num_leaves':512,'cat_smooth':1}
      t=time.perf_counter(); m=lgb.train(p,lgb.Dataset(hr(Xtr),label=ytr,categorical_feature=CAT),R2,valid_sets=[lgb.Dataset(hr(Xte),label=yte,categorical_feature=CAT)],callbacks=[lgb.record_evaluation(ev:={})] ,feval=None)
      for k in [100,200,300,500,700,1000]: print('ceiling R=%d lr=%.2f at %d: AUC %.4f'%(R2,lr,k,roc_auc_score(yte,m.predict(hr(Xte),num_iteration=k))),flush=True)
  if e=='linear':
    run('linear_tree',hr(Xtr),hr(Xte),CAT,linear_tree=True,cat_smooth=1)
  if e=='smooth':
    for cs in [0.1,0.5,1,2,3]: run(f'hour cat_smooth {cs}',hr(Xtr),hr(Xte),CAT,cat_smooth=cs)
  if e=='mdl':
    for v in [20,50,200,500]: run(f'hour cs1 min_data_in_leaf {v}',hr(Xtr),hr(Xte),CAT,cat_smooth=1,min_data_in_leaf=v)
  if e=='l2':
    for v in [1,10,100]: run(f'hour cs1 lambda_l2 {v}',hr(Xtr),hr(Xte),CAT,cat_smooth=1,lambda_l2=v)
  if e=='leaves':
    for nl in [2048,4096,8192]: run(f'hour cs1 leaves {nl}',hr(Xtr),hr(Xte),CAT,cat_smooth=1,num_leaves=nl)
  if e=='leaves_d':
    for nl,d in [(1024,10),(1024,14),(1024,20)]: run(f'hour cs1 leaves {nl} depth {d}',hr(Xtr),hr(Xte),CAT,cat_smooth=1,num_leaves=nl,max_depth=d)
