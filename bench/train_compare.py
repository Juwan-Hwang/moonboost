"""Train with moonboost (CLI) and XGBoost (hist, 1 thread) on the first 15,000
rows of bench/train_data.py output, score both on the rest, and check that
XGBoost predicts the moonboost model bit for bit. Usage:
    python bench/train_data.py && python bench/train_compare.py [rounds]"""
import numpy as np, xgboost as xgb, subprocess, time, json, sys
import os
B=os.environ.get('MOONBOOST', '_build/native/release/build/cmd/main/main.exe')
X=np.loadtxt('/tmp/tr_X.csv',delimiter=',',dtype=np.float32)
ntr=15000
np.savetxt('/tmp/trn_X.csv',X[:ntr],delimiter=',',fmt='%.9g'); np.savetxt('/tmp/tst_X.csv',X[ntr:],delimiter=',',fmt='%.9g')
R=int(sys.argv[1]) if len(sys.argv)>1 else 100
for nm,obj,extra,xp in [('r','reg:squarederror',[],{}),('b','binary:logistic',[],{}),('c','multi:softprob',['--num-class','3'],{'num_class':3})]:
  y=np.loadtxt(f'/tmp/tr_y{nm}.csv',dtype=np.float32)
  np.savetxt('/tmp/trn_y.csv',y[:ntr],fmt='%.9g')
  t=time.perf_counter(); out=subprocess.run([B,'train','/tmp/trn_X.csv','/tmp/trn_y.csv','-o','/tmp/mb.json','--rounds',str(R),'--objective',obj]+extra,capture_output=True,text=True).stdout.strip()
  mbms=float(out.split(' in ')[1].split()[0])
  mb=xgb.Booster(model_file='/tmp/mb.json')
  dte=xgb.DMatrix(X[ntr:]); dtr=xgb.DMatrix(X[:ntr],label=y[:ntr])
  p_mb_x=mb.predict(dte)
  cli=subprocess.run([B,'predict','/tmp/mb.json','/tmp/tst_X.csv'],capture_output=True,text=True).stdout
  p_mb=np.array([[float(v) for v in l.split(',')] for l in cli.strip().splitlines()],dtype=np.float32).reshape(p_mb_x.shape)
  same=np.sum(p_mb.view(np.uint32)==p_mb_x.view(np.uint32))
  params={'objective':obj,'tree_method':'hist','max_depth':6,'eta':0.3,'max_bin':256,'nthread':1,**xp}
  t=time.perf_counter(); xg=xgb.train(params,dtr,R); xgms=(time.perf_counter()-t)*1000
  p_xg=xg.predict(dte); yt=y[ntr:]
  def metric(p):
    if nm=='r': return 'rmse %.4f'%np.sqrt(np.mean((p-yt)**2))
    if nm=='b': p=np.clip(p,1e-7,1-1e-7); return 'logloss %.4f'%(-np.mean(yt*np.log(p)+(1-yt)*np.log(1-p)))
    return 'mlogloss %.4f acc %.4f'%(-np.mean(np.log(np.clip(p[np.arange(len(yt)),yt.astype(int)],1e-7,1))), np.mean(p.argmax(1)==yt))
  print(nm, f'moonboost {mbms:.0f} ms {metric(p_mb)} | xgboost {xgms:.0f} ms {metric(p_xg)} | xgb-on-our-model same bits {same}/{p_mb.size}')
