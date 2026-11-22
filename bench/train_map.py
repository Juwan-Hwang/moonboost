# Run in the directory of bench/train_rank.py's MSLR-WEB10K files (X_*.npy, y_*.npy, g_*.npy, Xtr.csv, gtr.txt), MOONBOOST=path to the CLI.
# MSLR-WEB10K fold 1, labels binarised (relevance >= 1): rank:map with moonboost and XGBoost, LightGBM lambdarank for reference; one thread
import numpy as np, subprocess, os, sys, time, json, re, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST'); R=int(os.environ.get('ROUNDS','100')); L=int(os.environ.get('LEAVES','255')); ETA=0.1
Xtr,ytr,gtr=np.load('X_train.npy'),(np.load('y_train.npy')>=1).astype(np.float32),np.load('g_train.npy')
Xte,yte,gte=np.load('X_test.npy'),(np.load('y_test.npy')>=1).astype(np.float32),np.load('g_test.npy')
if not os.path.exists('ytr_bin.csv'): np.savetxt('ytr_bin.csv',ytr,fmt='%d')
def qap(p,y,g):
    s=[];o=0
    for c in g:
        pp=p[o:o+c]; l=y[o:o+c]; o+=c
        if l.sum()==0: continue
        idx=np.argsort(-pp,kind='stable'); r=l[idx]; hits=np.cumsum(r)
        s.append((r*hits/np.arange(1,c+1)).sum()/r.sum())
    return np.array(s)
dte=xgb.DMatrix(Xte); res={}; q={}
if B:
    r=subprocess.run([B,'train','Xtr.csv','ytr_bin.csv','-o','/tmp/rmap_mb.json','--objective','rank:map','--group','gtr.txt','--rounds',str(R),'--eta',str(ETA),'--grow-policy','lossguide','--depth','0','--max-leaves',str(L),'--min-child-weight','0.001','--min-data-in-leaf','20'],capture_output=True,text=True)
    if r.returncode: print(r.stdout,r.stderr); sys.exit(1)
    tm=int(re.search(r'in (\d+) ms',r.stdout+r.stderr)[1])/1000
    b=xgb.Booster(model_file='/tmp/rmap_mb.json'); pm=b.predict(dte); q['mb']=qap(pm,yte,gte); res['mb']=(tm,q['mb'].mean())
    if os.environ.get('BIT')=='1':
        import subprocess as sp
        o=sp.run([B,'predict','/tmp/rmap_mb.json','Xte.csv'],capture_output=True,text=True).stdout.split()
        print('bitwise', np.array_equal(np.array(o,dtype=np.float32),pm), flush=True)
if os.environ.get('RIVALS','1')=='1':
    t=time.time(); d=xgb.DMatrix(Xtr,ytr,nthread=1); d.set_group(gtr); b=xgb.train({'objective':'rank:map','eta':ETA,'tree_method':'hist','grow_policy':'lossguide','max_depth':0,'max_leaves':L,'min_child_weight':0.001,'nthread':1},d,R); tx=time.time()-t
    px=b.predict(dte); q['xgb']=qap(px,yte,gte); res['xgb']=(tx,q['xgb'].mean())
    t=time.time(); ds=lgb.Dataset(Xtr,ytr,group=gtr); m=lgb.train({'objective':'lambdarank','learning_rate':ETA,'num_leaves':L,'min_sum_hessian_in_leaf':0.001,'num_threads':1,'verbose':-1},ds,R); tl=time.time()-t
    pl=m.predict(Xte); q['lgb']=qap(pl,yte,gte); res['lgb']=(tl,q['lgb'].mean())
if 'mb' in q:
    for k in q:
        if k!='mb': d=q['mb']-q[k]; print(f'mb-{k} {d.mean():+.4f}±{d.std()/len(d)**.5:.4f}',end=' ')
print(' '.join(f'{k} {v[0]:.1f}s MAP {v[1]:.4f}' for k,v in res.items()), flush=True)
