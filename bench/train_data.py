"""Synthetic training data for bench/train_compare.py: 20,000 rows, 8 features,
5% missing; regression, binary and 3-class labels, written to /tmp/tr_*.csv."""
import numpy as np, sys
rng=np.random.default_rng(0)
n=20000; X=rng.normal(size=(n,8)).astype(np.float32)
X[rng.random(X.shape)<0.05]=np.nan
f=np.nan_to_num(X)
yr=(np.sin(f[:,0])*2+f[:,1]*f[:,2]+0.5*f[:,3]+rng.normal(scale=0.3,size=n)).astype(np.float32)
yb=(yr>0.3).astype(np.float32)
yc=np.digitize(yr,[-1,1]).astype(np.float32)
np.savetxt('/tmp/tr_X.csv',X,delimiter=',',fmt='%.9g')
for nm,y in [('r',yr),('b',yb),('c',yc)]: np.savetxt(f'/tmp/tr_y{nm}.csv',y,fmt='%.9g')
