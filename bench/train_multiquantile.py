# reg:quantileerror with several quantiles in one model (quantile_alpha [0.1, 0.5, 0.9]) on 100,000 synthetic rows with
# heavy-tailed, heteroscedastic noise (80,000 to train), four seeds: moonboost and XGBoost each fit one three-output model,
# LightGBM (which has no multi-quantile objective) three models. XGBoost loads our model and must predict the same bits.
# Reports wall time of each fit (one thread, data loading included) and held-out pinball loss per quantile.
import numpy as np, subprocess, os, sys, time, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST','moonboost'); EX=sys.argv[1:]; W='/tmp/objmq'; os.makedirs(W,exist_ok=True)
A=[0.1,0.5,0.9]; R=100
for seed in [0,1,2,3]:
  r=np.random.RandomState(seed); n=100000; X=r.rand(n,8).astype(np.float32); X[r.rand(n,8)<0.05]=np.nan
  f=np.nan_to_num(np.sin(6*X[:,0])+2*X[:,1]*X[:,2]-X[:,3]+0.5*(X[:,4]>.5))
  y=np.nan_to_num((f+r.standard_t(2,n)*(0.5+np.nan_to_num(X[:,5]))).astype(np.float32)); k=80000
  np.savetxt(f'{W}/X.csv',X[:k],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[k:],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/y.csv',y[:k],fmt='%.9g')
  t=time.time(); subprocess.run([B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective','reg:quantileerror','--quantile-alpha',','.join(map(str,A)),'--rounds',str(R),'--eta','0.1','--depth','6']+EX,check=True,capture_output=True); tm=time.time()-t
  pm=np.array(subprocess.run([B,'predict',f'{W}/m.json',f'{W}/t.csv'],capture_output=True,text=True).stdout.replace(',',' ').split(),dtype=np.float32).reshape(-1,len(A))
  px=xgb.Booster(model_file=f'{W}/m.json').predict(xgb.DMatrix(X[k:])).astype(np.float32)
  same=np.array_equal(pm.view(np.uint32),px.view(np.uint32))
  t=time.time(); xb=xgb.train({'objective':'reg:quantileerror','quantile_alpha':A,'eta':0.1,'max_depth':6,'tree_method':'hist','nthread':1},xgb.DMatrix(X[:k],label=y[:k]),R); tx=time.time()-t
  Px=xb.predict(xgb.DMatrix(X[k:]))
  t=time.time(); Pl=np.stack([lgb.train({'objective':'quantile','alpha':a,'learning_rate':0.1,'max_depth':6,'num_leaves':63,'verbose':-1,'num_threads':1},lgb.Dataset(X[:k],label=y[:k]),R).predict(X[k:]) for a in A],1); tl=time.time()-t
  out=[]
  for j,a in enumerate(A):
    L=lambda p:np.maximum(a*(y[k:]-p),(a-1)*(y[k:]-p)); dx=L(pm[:,j])-L(Px[:,j]); dl=L(pm[:,j])-L(Pl[:,j])
    out.append(f'a={a} mb {L(pm[:,j]).mean():.5f} xgb {L(Px[:,j]).mean():.5f} lgb {L(Pl[:,j]).mean():.5f} dX {dx.mean():+.5f}±{dx.std()/len(dx)**.5:.5f} dL {dl.mean():+.5f}±{dl.std()/len(dl)**.5:.5f}')
  print(f'seed {seed} bitwise {same} time mb {tm:.2f}s xgb {tx:.2f}s lgb(3 models) {tl:.2f}s | '+' | '.join(out),flush=True)
