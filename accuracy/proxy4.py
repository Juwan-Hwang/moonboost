# LightGBM proxies at 1024 leaves: lr schedules (N5), path smoothing (N8), extra_trees
import numpy as np, time, sys, os, json, lightgbm as lgb
from sklearn.metrics import roc_auc_score
Xtr=np.load('/tmp/Xtr.npy'); Xte=np.load('/tmp/Xte.npy'); ytr=np.load('/tmp/ytr.npy'); yte=np.load('/tmp/yte.npy')
os.sched_setaffinity(0,{int(os.environ.get('CORE',2))})
CAT=[0,1,2,5,6,7]; hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]; A,B=hr(Xtr),hr(Xte)
def run(name,cb=None,**kw):
  p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1,'num_leaves':1024}; p.update(kw)
  t=time.perf_counter(); m=lgb.train(p,lgb.Dataset(A,label=ytr,categorical_feature=CAT),100,callbacks=cb or [])
  print('%-40s %6.1fs AUC %.4f'%(name,time.perf_counter()-t,roc_auc_score(yte,m.predict(B))),flush=True)
for e in sys.argv[1:]:
  if e=='base': run('base hour leaves1024')
  if e=='path':
    for v in [1,10,100]: run(f'path_smooth {v}',path_smooth=v)
  if e=='sched':
    run('lr linear 0.2->0.02',cb=[lgb.reset_parameter(learning_rate=lambda i:0.2-0.18*i/99)])
    run('lr cosine 0.15->0.03',cb=[lgb.reset_parameter(learning_rate=lambda i:0.03+0.06*(1+np.cos(np.pi*i/99)))])
  if e=='extra': run('extra_trees',extra_trees=True)
  if e=='onehot': run('max_cat_to_onehot 32',max_cat_to_onehot=32)
  if e=='mcg': 
    for v in [10,50,300]: run(f'min_data_per_group {v}',min_data_per_group=v)
  if e=='mct':
    for v in [8,64,256]: run(f'max_cat_threshold {v}',max_cat_threshold=v)
  if e=='oh':
    for v in [8,16,64,512]: run(f'max_cat_to_onehot {v}',max_cat_to_onehot=v)
  if e=='oh2':
    run('onehot32 + mct256',max_cat_to_onehot=32,max_cat_threshold=256)
    run('onehot32 + leaves2048',max_cat_to_onehot=32,num_leaves=2048)
    run('onehot32 nohour?',max_cat_to_onehot=32,num_leaves=1024)
  if e=='lgbbig':
    run('oh32 leaves4096',max_cat_to_onehot=32,num_leaves=4096)
    run('oh32 leaves3072',max_cat_to_onehot=32,num_leaves=3072)
  if e=='lgbmatch':
    for nl in [2048,4096]: run(f'oh32 leaves{nl} lambda1 mshl1 mdl0',max_cat_to_onehot=32,num_leaves=nl,lambda_l2=1,min_sum_hessian_in_leaf=1,min_data_in_leaf=1)
