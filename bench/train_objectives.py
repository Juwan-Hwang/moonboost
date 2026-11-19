# New training objectives: XGBoost loads our model and predicts bit-for-bit what
# moonboost predicts; held-out loss vs XGBoost and LightGBM with the same settings.
import numpy as np, subprocess, os, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST', 'moonboost'); W='/tmp/objc'; os.makedirs(W,exist_ok=True)
r=np.random.RandomState(0); n=20000; X=r.rand(n,8).astype(np.float32); X[r.rand(n,8)<0.05]=np.nan
f=np.nan_to_num(np.sin(6*X[:,0])+2*X[:,1]*X[:,2]-X[:,3]+0.5*(X[:,4]>.5))
y={'count:poisson':r.poisson(np.exp(f)),'reg:gamma':r.gamma(2.0,np.exp(f)/2.0),
   'reg:tweedie':r.poisson(np.exp(f))*r.gamma(2.0,0.5,n),'reg:pseudohubererror':f+r.standard_t(2,n),
   'reg:squaredlogerror':np.expm1(np.abs(f)+0.3*r.rand(n)),'reg:logistic':1/(1+np.exp(-f)),'binary:logitraw':(r.rand(n)<1/(1+np.exp(-f))).astype(float)}
lgbname={'count:poisson':'poisson','reg:gamma':'gamma','reg:tweedie':'tweedie','reg:pseudohubererror':'huber','reg:squaredlogerror':None,'reg:logistic':'cross_entropy','binary:logitraw':'binary'}
def loss(o,yt,p):
  if o=='count:poisson': return np.mean(p-yt*np.log(p))
  if o=='reg:gamma': return np.mean(yt/p+np.log(p))
  if o=='reg:tweedie': rho=1.5; return np.mean(-yt*p**(1-rho)/(1-rho)+p**(2-rho)/(2-rho))
  if o=='reg:pseudohubererror': z=p-yt; return np.mean(np.sqrt(1+z*z)-1)
  if o=='reg:squaredlogerror': return np.sqrt(np.mean((np.log1p(np.maximum(p,-1+1e-6))-np.log1p(yt))**2))
  if o=='reg:logistic': p=np.clip(p,1e-7,1-1e-7); return -np.mean(yt*np.log(p)+(1-yt)*np.log(1-p))
  if o=='binary:logitraw': p=1/(1+np.exp(-p)); p=np.clip(p,1e-7,1-1e-7); return -np.mean(yt*np.log(p)+(1-yt)*np.log(1-p))
k=16000
np.savetxt(f'{W}/X.csv',X[:k],delimiter=',',fmt='%.9g'); np.savetxt(f'{W}/t.csv',X[k:],delimiter=',',fmt='%.9g')
for o,yy in y.items():
  yy=yy.astype(np.float32); np.savetxt(f'{W}/y.csv',yy[:k],fmt='%.9g')
  subprocess.run([B,'train',f'{W}/X.csv',f'{W}/y.csv','-o',f'{W}/m.json','--objective',o,'--rounds','100','--eta','0.1','--depth','6'],check=True,capture_output=True)
  pm=np.array(subprocess.run([B,'predict',f'{W}/m.json',f'{W}/t.csv'],capture_output=True,text=True).stdout.split(),dtype=np.float32)
  bst=xgb.Booster(model_file=f'{W}/m.json'); px=bst.predict(xgb.DMatrix(X[k:]))
  same=np.array_equal(pm.view(np.uint32),px.astype(np.float32).view(np.uint32))
  xb=xgb.train({'objective':o,'eta':0.1,'max_depth':6,'tree_method':'hist','nthread':1},xgb.DMatrix(X[:k],label=yy[:k]),100)
  lx=loss(o,yy[k:],xb.predict(xgb.DMatrix(X[k:])))
  ll='-'
  if lgbname[o]:
    lm=lgb.train({'objective':lgbname[o],'learning_rate':0.1,'max_depth':6,'num_leaves':63,'verbose':-1,'num_threads':1},lgb.Dataset(X[:k],label=yy[:k]),100)
    pl=lm.predict(X[k:],raw_score=(o=='binary:logitraw')); ll='%.5f'%loss(o,yy[k:],pl)
  print(f'{o:22s} bitwise {same}  loss mb {loss(o,yy[k:],pm):.5f}  xgb {lx:.5f}  lgb {ll}',flush=True)
