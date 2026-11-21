# rank:ndcg on MSLR-WEB10K fold 1 (723,412 training rows in 6,000 queries, 136 features; test: 2,000 queries)
# against XGBoost rank:ndcg and LightGBM lambdarank: one thread, 100 rounds, eta 0.1, 255 leaves, min_child_weight
# 0.001; NDCG@10 on the test queries (a query with no relevant document counts as 1), with the per-query
# difference and its standard error. Data: the parquet files of huggingface.co/datasets/philipphager/MSLR-WEB10k,
# converted by `python bench/train_rank.py prep` in the directory holding them. MOONBOOST names the CLI binary.
import sys
if sys.argv[1:2]==['prep']:
    import pandas as pd, numpy as np
    for sp in ['train','test','validation']:
        d=pd.read_parquet(f'{sp}.parquet')
        X=np.concatenate([np.asarray(np.stack(f),dtype=np.float32) for f in d['features']])
        y=np.concatenate([np.asarray(l,dtype=np.float32) for l in d['labels']])
        g=np.array([len(l) for l in d['labels']])
        print(sp,X.shape,y.shape,g.shape,np.bincount(y.astype(int)))
        np.save(f'X_{sp}.npy',X); np.save(f'y_{sp}.npy',y); np.save(f'g_{sp}.npy',g)
        if sp=='train':
            np.savetxt('Xtr.csv',X,delimiter=',',fmt='%.9g'); np.savetxt('ytr.csv',y,fmt='%d'); np.savetxt('gtr.txt',g,fmt='%d')
    sys.exit()
import numpy as np, subprocess, os, sys, time, json, xgboost as xgb, lightgbm as lgb
B=os.environ.get('MOONBOOST'); EX=sys.argv[1:]
R=int(os.environ.get('ROUNDS','100')); L=int(os.environ.get('LEAVES','255')); ETA=0.1
Xtr,ytr,gtr=np.load('X_train.npy'),np.load('y_train.npy'),np.load('g_train.npy')
Xte,yte,gte=np.load('X_test.npy'),np.load('y_test.npy'),np.load('g_test.npy')
def ndcg(pred,y,g,k=10):
    return qndcg(pred,y,g,k).mean()
def qndcg(pred,y,g,k=10):
    s=[]; o=0
    for c in g:
        p=pred[o:o+c]; l=y[o:o+c]; o+=c
        gain=2.0**l-1; disc=1/np.log2(np.arange(2,c+2))
        ideal=(np.sort(gain)[::-1][:k]*disc[:min(k,c)]).sum()
        if ideal==0: s.append(1.0); continue
        idx=np.argsort(-p,kind='stable')[:k]
        s.append((gain[idx]*disc[:len(idx)]).sum()/ideal)
    return np.array(s)
dte=xgb.DMatrix(Xte)
res={}
if B:
    t=time.time(); r=subprocess.run([B,'train','Xtr.csv','ytr.csv','-o','/tmp/rank_mb.json','--objective','rank:ndcg','--group','gtr.txt','--rounds',str(R),'--eta',str(ETA),'--grow-policy','lossguide','--depth','0','--max-leaves',str(L),'--min-child-weight','0.001','--min-data-in-leaf','20',*EX],capture_output=True,text=True)
    if r.returncode: print(r.stderr); sys.exit(1)
    import re; tm=int(re.search(r'in (\d+) ms',r.stdout+r.stderr)[1])/1000
    b=xgb.Booster(model_file='/tmp/rank_mb.json')
    pm=b.predict(dte); res['mb']=(tm,ndcg(pm,yte,gte)); qm=qndcg(pm,yte,gte)
if os.environ.get('RIVALS','1')=='1':
    d=xgb.DMatrix(Xtr,ytr); d.set_group(gtr)
    t=time.time(); d=xgb.DMatrix(Xtr,ytr,nthread=1); d.set_group(gtr); b=xgb.train({'objective':'rank:ndcg','eta':ETA,'tree_method':'hist','grow_policy':'lossguide','max_depth':0,'max_leaves':L,'min_child_weight':0.001,'nthread':1},d,R); tx=time.time()-t
    px=b.predict(dte); res['xgb']=(tx,ndcg(px,yte,gte))
    t=time.time(); ds=lgb.Dataset(Xtr,ytr,group=gtr); m=lgb.train({'objective':'lambdarank','learning_rate':ETA,'num_leaves':L,'min_sum_hessian_in_leaf':0.001,'num_threads':1,'verbose':-1},ds,R); tl=time.time()-t
    pl=m.predict(Xte); res['lgb']=(tl,ndcg(pl,yte,gte))
if B and 'lgb' in res:
    for nm,pp in [('xgb',px),('lgb',pl)]:
        d=qm-qndcg(pp,yte,gte); print(f'mb-{nm} {d.mean():+.4f}±{d.std()/len(d)**.5:.4f}',end=' ')
print(' '.join(f'{k} {v[0]:.1f}s ndcg@10 {v[1]:.4f}' for k,v in res.items()), EX, flush=True)
