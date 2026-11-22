# binary:hinge on GBM-perf airline (X.csv/y.csv as for the other GBM-perf benches): moonboost CLI vs XGBoost, one thread
import numpy as np, subprocess, time, xgboost as xgb, re, json, os
B=os.environ['MOONBOOST']
X=np.loadtxt(os.environ.get('X','X.csv'),delimiter=',',dtype=np.float32); y=np.loadtxt(os.environ.get('Y','y.csv'),dtype=np.float32)
n=800000; Xtr,ytr,Xte,yte=X[:n],y[:n],X[n:],y[n:]
np.savetxt('/tmp/h_X.csv',Xtr,delimiter=',',fmt='%.9g'); np.savetxt('/tmp/h_y.csv',ytr,fmt='%g'); np.savetxt('/tmp/h_Xt.csv',Xte[:20000],delimiter=',',fmt='%.9g')
for depth,R in [(6,100),(10,100)]:
  r=subprocess.run([B,'train','/tmp/h_X.csv','/tmp/h_y.csv','-o','/tmp/h_m.json','--objective','binary:hinge','--rounds',str(R),'--depth',str(depth),'--eta','0.1'],capture_output=True,text=True)
  tm=int(re.search(r'in (\d+) ms',r.stdout+r.stderr)[1])/1000
  b=xgb.Booster(model_file='/tmp/h_m.json'); pm=b.predict(xgb.DMatrix(Xte))
  mm=b.predict(xgb.DMatrix(Xte[:20000]),output_margin=True)
  q=np.array(subprocess.run([B,'predict','/tmp/h_m.json','/tmp/h_Xt.csv','--margin'],capture_output=True,text=True).stdout.split(),dtype=np.float32)
  bit=bool((mm.view(np.uint32)==q.view(np.uint32)).all())
  t=time.time(); bx=xgb.train({'objective':'binary:hinge','eta':0.1,'max_depth':depth,'tree_method':'hist','nthread':1},xgb.DMatrix(Xtr,ytr,nthread=1),R); tx=time.time()-t
  px=bx.predict(xgb.DMatrix(Xte))
  print(f'depth{depth} mb {tm:.1f}s acc {(pm==yte).mean():.4f} bitwise {bit} | xgb {tx:.1f}s acc {(px==yte).mean():.4f} base mb {json.loads(open("/tmp/h_m.json").read())["learner"]["learner_model_param"]["base_score"]} xgb {json.loads(bx.save_raw("json"))["learner"]["learner_model_param"]["base_score"]}',flush=True)
