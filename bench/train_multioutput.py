# Several outputs in one model (num_target 3) on 250,000 synthetic rows, 10 features (200,000 to train), three seeds:
# reg:squarederror on three regression targets and binary:logistic on three labels. moonboost and XGBoost (its default
# one_output_per_tree) each fit one three-output model, LightGBM three models. XGBoost loads our model and must predict
# the same bits. Wall time of each fit (one thread, data loading included); held-out RMSE or log loss per output.
import numpy as np, subprocess, os, sys, time, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST','moonboost'); EX=sys.argv[1:]; W='/tmp/objmo'; os.makedirs(W,exist_ok=True); R=100
for seed in [0,1,2]:
  r=np.random.RandomState(seed); n=250000; X=r.rand(n,10).astype(np.float32); X[r.rand(n,10)<0.05]=np.nan; Z=np.nan_to_num(X,nan=0.5)
  F=np.stack([np.sin(6*Z[:,0])+2*Z[:,1]*Z[:,2], Z[:,3]**2-Z[:,4]+(Z[:,5]>.5), np.cos(4*Z[:,6]*Z[:,7])+Z[:,8]-Z[:,9]],1)
  k=200000
  np.savetxt(f'{W}/X.csv',X[:k],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[k:],delimiter=',',fmt='%.9g')
  for o in ['reg:squarederror','binary:logistic']:
    Y=(F+r.normal(size=F.shape)*0.5).astype(np.float32)
    if o=='binary:logistic': Y=(Y>np.median(Y,0)).astype(np.float32)
    np.savetxt(f'{W}/y.csv',Y[:k],delimiter=',',fmt='%.9g')
    t=time.time(); subprocess.run([B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective',o,'--rounds',str(R),'--eta','0.1','--depth','6']+EX,check=True,capture_output=True); tm=time.time()-t
    pm=np.array(subprocess.run([B,'predict',f'{W}/m.json',f'{W}/t.csv'],capture_output=True,text=True).stdout.replace(',',' ').split(),dtype=np.float32).reshape(-1,3)
    px=xgb.Booster(model_file=f'{W}/m.json').predict(xgb.DMatrix(X[k:])).astype(np.float32)
    same=np.array_equal(pm.view(np.uint32),px.view(np.uint32))
    t=time.time(); xb=xgb.train({'objective':o,'eta':0.1,'max_depth':6,'tree_method':'hist','nthread':1},xgb.DMatrix(X[:k],label=Y[:k]),R); tx=time.time()-t
    Px=xb.predict(xgb.DMatrix(X[k:]))
    lo='regression' if o=='reg:squarederror' else 'binary'
    t=time.time(); Pl=np.stack([lgb.train({'objective':lo,'learning_rate':0.1,'max_depth':6,'num_leaves':63,'verbose':-1,'num_threads':1},lgb.Dataset(X[:k],label=Y[:k,j]),R).predict(X[k:]) for j in range(3)],1); tl=time.time()-t
    def loss(P,j):
      y=Y[k:,j].astype(np.float64); p=P[:,j].astype(np.float64)
      if o=='reg:squarederror': return (p-y)**2
      p=np.clip(p,1e-7,1-1e-7); return -(y*np.log(p)+(1-y)*np.log(1-p))
    out=[]
    for j in range(3):
      a,b,c=loss(pm,j),loss(Px,j),loss(Pl,j); dx=a-b; dl=a-c
      f=(lambda v:np.sqrt(v.mean())) if o=='reg:squarederror' else (lambda v:v.mean())
      out.append(f'y{j} mb {f(a):.5f} xgb {f(b):.5f} lgb {f(c):.5f} dX {dx.mean():+.5f}±{dx.std()/len(dx)**.5:.5f} dL {dl.mean():+.5f}±{dl.std()/len(dl)**.5:.5f}')
    print(f'seed {seed} {o} bitwise {same} time mb {tm:.2f}s xgb {tx:.2f}s lgb(3) {tl:.2f}s | '+' | '.join(out),flush=True)
