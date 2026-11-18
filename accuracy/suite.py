# binary-classification benchmark suite from OpenML: saves /data/home/mb/acc/ds/<name>.npz
import numpy as np, os, sys
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
OUT=os.path.expanduser('~/mb/acc/ds'); os.makedirs(OUT,exist_ok=True)
SETS={'higgs':23512,'miniboone':41150,'adult':1590,'amazon':4135,'bank':1461,'electricity':151,'kdd_upselling':1114,'credit':31,'phoneme':1489,'albert':41147,'click':1220,'jannis':41168}
for retry in (False,True):
 for name,i in SETS.items():
   p=f'{OUT}/{name}.npz'
   if os.path.exists(p): continue
   try:
     d=fetch_openml(data_id=i,as_frame=True,parser=('liac-arff' if retry else 'auto')); X=d.data; y=d.target
     cats=[j for j,c in enumerate(X.columns) if str(X[c].dtype) in ('category','object')]
     A=np.empty(X.shape,np.float32)
     for j,c in enumerate(X.columns):
       if j in cats: A[:,j]=X[c].astype('category').cat.codes.replace(-1,np.nan) if False else np.where(X[c].isna(),np.nan,X[c].astype('category').cat.codes).astype(np.float32)
       else: A[:,j]=X[c].astype(np.float32)
     yv=y.astype('category').cat.codes.values
     if len(np.unique(yv))!=2: print(name,'not binary',np.unique(yv)); continue
     tr,te=train_test_split(np.arange(len(yv)),test_size=0.2,random_state=0,stratify=yv)
     np.savez(p,Xtr=A[tr],Xte=A[te],ytr=yv[tr].astype(np.float32),yte=yv[te],cat=np.array(cats,dtype=np.int64))
     print(name,A.shape,'cats',len(cats),flush=True)
   except Exception as e: print(name,'fail',e,flush=True)
