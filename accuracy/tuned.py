# Equal-budget tuning benchmark: each library gets the same number of random configs
# (100 rounds, lr 0.1, one thread), picks by validation AUC, reports test AUC.
import numpy as np, subprocess, os, sys, time, json
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
core=os.environ.get('CORE','0'); os.sched_setaffinity(0,{int(core)})
TRIALS=int(os.environ.get('TRIALS','24'))
B=os.path.expanduser('~/mb/moonboost/_build/native/release/build/cmd/main/main.exe')
def logu(r,a,b): return float(np.exp(r.uniform(np.log(a),np.log(b))))
def shared(r):
  # one draw, mapped to each library's names, so all three see the same settings
  c={'lossguide':r.rand()<0.5,'depth':int(r.randint(3,11)),'leaves':int(logu(r,15,4096)),'lam':logu(r,0.1,30),'mcw':logu(r,0.1,30),
     'onehot':int(r.choice([4,32])),'mdpg':int(r.choice([1,10,100])),'cat_l2':float(r.choice([1,10])),'cat_smooth':float(r.choice([1,10,30])),
     'mct':int(r.choice([16,64,256])),'min_data':int(r.choice([1,20,100]))}
  return c
def mb_cfg(c):
  a=['--grow-policy','lossguide','--depth','0','--max-leaves',str(c['leaves'])] if c['lossguide'] else ['--depth',str(c['depth'])]
  return a+['--lambda','%.4g'%c['lam'],'--min-child-weight','%.4g'%c['mcw'],'--max-cat-to-onehot',str(c['onehot']),'--min-data-per-group',str(c['mdpg']),
            '--cat-l2',str(c['cat_l2']),'--cat-smooth',str(c['cat_smooth']),'--max-cat-threshold',str(c['mct'])]
def lgb_cfg(c):
  d={'num_leaves':c['leaves'],'max_depth':-1} if c['lossguide'] else {'num_leaves':min(2**c['depth'],131072),'max_depth':c['depth']}
  d.update(lambda_l2=c['lam'],min_sum_hessian_in_leaf=c['mcw'],min_data_in_leaf=c['min_data'],max_cat_to_onehot=c['onehot'],min_data_per_group=c['mdpg'],
           cat_l2=c['cat_l2'],cat_smooth=c['cat_smooth'],max_cat_threshold=c['mct'])
  return d
def xgb_cfg(c):
  d={'grow_policy':'lossguide','max_depth':0,'max_leaves':c['leaves']} if c['lossguide'] else {'max_depth':c['depth']}
  d.update({'lambda':c['lam'],'min_child_weight':c['mcw'],'max_cat_to_onehot':c['onehot'],'max_cat_threshold':c['mct']})
  return d
def run(lib,cfg,Xf,yf,Xs,cats,W):
  if lib=='moonboost':
    extra=['--categorical',','.join(map(str,cats))] if len(cats) else []
    o=subprocess.run(['taskset','-c',core,B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective','binary:logistic','--rounds','100','--eta','0.1']+extra+cfg,capture_output=True,text=True)
    if o.returncode: raise RuntimeError(o.stdout[-200:]+o.stderr[-200:])
    return [np.array(subprocess.run([B,'predict',f'{W}/m.json',f],capture_output=True,text=True).stdout.split(),dtype=float) for f in (f'{W}/v.csv',f'{W}/t.csv')]
  if lib=='lightgbm':
    p={'objective':'binary','learning_rate':0.1,'num_threads':1,'verbose':-1}; p.update(cfg)
    m=lgb.train(p,lgb.Dataset(Xf,label=yf,categorical_feature=[int(c) for c in cats]),100); return [m.predict(X) for X in Xs]
  ft=['c' if j in set(int(c) for c in cats) else 'q' for j in range(Xf.shape[1])]
  p={'objective':'binary:logistic','eta':0.1,'tree_method':'hist','nthread':1}; p.update(cfg)
  m=xgb.train(p,xgb.DMatrix(Xf,label=yf,feature_types=ft,enable_categorical=True,nthread=1),100)
  return [m.predict(xgb.DMatrix(X,feature_types=ft,enable_categorical=True)) for X in Xs]
CFG={'moonboost':mb_cfg,'lightgbm':lgb_cfg,'xgboost':xgb_cfg}
libs=os.environ.get('LIBS','moonboost,lightgbm,xgboost').split(',')
for n in sys.argv[1:]:
  d=np.load(os.path.expanduser(f'~/mb/acc/ds/{n}.npz')); Xtr,Xte,ytr,yte,cats=d['Xtr'],d['Xte'],d['ytr'],d['yte'],d['cat']
  rng=np.random.RandomState(1); idx=rng.permutation(len(ytr)); k=int(len(ytr)*0.8); fi,vi=idx[:k],idx[k:]
  W=os.path.expanduser(f'~/mb/acc/tw_{n}_{core}'); os.makedirs(W,exist_ok=True)
  np.savetxt(f'{W}/X.csv',Xtr[fi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',ytr[fi],fmt='%g')
  np.savetxt(f'{W}/v.csv',Xtr[vi],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',Xte,delimiter=',',fmt='%.9g')
  for lib in libs:
    r=np.random.RandomState(42); best=None; t=time.perf_counter(); fails=0
    for i in range(TRIALS):
      cfg=CFG[lib](shared(r))
      try: pv,pt=run(lib,cfg,Xtr[fi],ytr[fi],[Xtr[vi],Xte],cats,W)
      except Exception as e: fails+=1; continue
      v=roc_auc_score(ytr[vi],pv)
      if best is None or v>best[0]: best=(v,roc_auc_score(yte,pt),cfg)
    print('%-14s %-10s valid %.4f test %.4f fails %d (%.0fs) %s'%(n,lib,best[0],best[1],fails,time.perf_counter()-t,best[2]),flush=True)
    with open(os.path.expanduser('~/mb/acc/tuned_results.jsonl'),'a') as f: f.write(json.dumps({'ds':n,'lib':lib,'valid':best[0],'test':best[1],'cfg':str(best[2]),'trials':TRIALS})+'\n')
