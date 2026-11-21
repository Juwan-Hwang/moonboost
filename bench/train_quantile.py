# reg:absoluteerror and reg:quantileerror on 20,000 synthetic rows with heavy-tailed noise, three seeds: XGBoost
# loads our model and predicts bit-for-bit what moonboost predicts; held-out pinball loss vs XGBoost and LightGBM.
# Extra arguments go to moonboost train (e.g. --min-data-in-leaf 20).
import numpy as np, subprocess, os, sys, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST','moonboost'); EX=sys.argv[1:]; W='/tmp/objq'; os.makedirs(W,exist_ok=True)
for seed in [0,1,2]:
  r=np.random.RandomState(seed); n=20000; X=r.rand(n,8).astype(np.float32); X[r.rand(n,8)<0.05]=np.nan
  f=np.nan_to_num(np.sin(6*X[:,0])+2*X[:,1]*X[:,2]-X[:,3]+0.5*(X[:,4]>.5))
  y=(f+r.standard_t(2,n)*(0.5+X[:,5].clip(0,1) if True else 1)).astype(np.float32); y=np.nan_to_num(y)
  k=16000
  np.savetxt(f'{W}/X.csv',X[:k],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[k:],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',y[:k],fmt='%.9g')
  for o,a in [('reg:absoluteerror',0.5),('reg:quantileerror',0.1),('reg:quantileerror',0.9)]:
    pin=lambda p:np.mean(np.maximum(a*(y[k:]-p),(a-1)*(y[k:]-p)))
    ex=['--quantile-alpha',str(a)] if o=='reg:quantileerror' else []
    subprocess.run([B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective',o,'--rounds','100','--eta','0.1','--depth','6']+ex+EX,check=True,capture_output=True)
    pm=np.array(subprocess.run([B,'predict',f'{W}/m.json',f'{W}/t.csv'],capture_output=True,text=True).stdout.split(),dtype=np.float32)
    px=xgb.Booster(model_file=f'{W}/m.json').predict(xgb.DMatrix(X[k:]))
    same=np.array_equal(pm.view(np.uint32),px.astype(np.float32).view(np.uint32))
    xp={'objective':o,'eta':0.1,'max_depth':6,'tree_method':'hist','nthread':1}
    if o=='reg:quantileerror': xp['quantile_alpha']=a
    xb=xgb.train(xp,xgb.DMatrix(X[:k],label=y[:k]),100)
    lm=lgb.train({'objective':'l1' if o=='reg:absoluteerror' else 'quantile','alpha':a,'learning_rate':0.1,'max_depth':6,'num_leaves':63,'verbose':-1,'num_threads':1},lgb.Dataset(X[:k],label=y[:k]),100)
    print(f'seed {seed} {o:18s} a={a}  bitwise {same}  loss mb {pin(pm):.5f}  xgb {pin(xb.predict(xgb.DMatrix(X[k:]))):.5f}  lgb {pin(lm.predict(X[k:])):.5f}',flush=True)
