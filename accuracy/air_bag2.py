import numpy as np, time, os, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
os.sched_setaffinity(0,{2})
Xtr=np.load("/tmp/Xtr.npy"); Xte=np.load("/tmp/Xte.npy"); ytr=np.load("/tmp/ytr.npy"); yte=np.load("/tmp/yte.npy")
hr=lambda X:np.c_[X,X[:,3]//100,X[:,3]%100]; A,B=hr(Xtr),hr(Xte); CAT=[0,1,2,5,6,7]
for L in [255,1023]:
  for bf in [1.0,0.5]:
    t=time.perf_counter(); m=lgb.train({"objective":"binary","learning_rate":0.1,"num_threads":1,"verbose":-1,"num_leaves":L,"max_cat_to_onehot":32,"bagging_fraction":bf,"bagging_freq":1 if bf<1 else 0},lgb.Dataset(A,label=ytr,categorical_feature=CAT),100)
    print("lightgbm L%d bag %.1f %.1fs AUC %.4f"%(L,bf,time.perf_counter()-t,roc_auc_score(yte,m.predict(B))),flush=True)
  ft=["c" if j in CAT else "q" for j in range(A.shape[1])]
  t=time.perf_counter(); m=xgb.train({"objective":"binary:logistic","eta":0.1,"tree_method":"hist","nthread":1,"grow_policy":"lossguide","max_depth":0,"max_leaves":L,"max_cat_to_onehot":32,"subsample":0.5},xgb.DMatrix(A,label=ytr,feature_types=ft,enable_categorical=True,nthread=1),100)
  el=time.perf_counter()-t
  print("xgboost L%d sub 0.5 %.1fs AUC %.4f"%(L,el,roc_auc_score(yte,m.predict(xgb.DMatrix(B,feature_types=ft,enable_categorical=True)))),flush=True)
