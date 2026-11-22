# Per-row held-out AFT negative log likelihood (scipy) of moonboost builds against XGBoost on bench/train_aft.py data.
# SEED=1 MOONBOOST=x python bench/train_aft_seeds.py logistic 0.5 4 path/to/moonboost [more builds]
import numpy as np, subprocess, xgboost as xgb, os, sys
exec(open('bench/train_aft.py').read().split("dte = xgb.DMatrix")[0].replace('default_rng(0)','default_rng(int(os.environ.get("SEED","0")))'))
B=os.environ['MOONBOOST']
dist,scale,depth=sys.argv[1],float(sys.argv[2]),int(sys.argv[3])
from scipy.stats import norm, logistic, gumbel_l
D={'normal':norm,'logistic':logistic,'extreme':gumbel_l}[dist]
def nll(pred):
    m=np.log(pred.astype(np.float64)); l=lo[ntr:].astype(np.float64); h=hi[ntr:].astype(np.float64)
    out=np.empty(len(m)); ob=l==h
    z=(np.log(l[ob])-m[ob])/scale; out[ob]=-(D.logpdf(z)-np.log(scale*l[ob]))
    cu=np.where(np.isinf(h[~ob]),1.0,D.cdf((np.log(np.where(np.isinf(h[~ob]),1,h[~ob]))-m[~ob])/scale))
    cl=np.where(l[~ob]<=0,0.0,D.cdf((np.log(np.maximum(l[~ob],1e-300))-m[~ob])/scale))
    out[~ob]=-np.log(np.maximum(cu-cl,1e-12)); return out
dte=xgb.DMatrix(X[ntr:]); dtr=xgb.DMatrix(X[:ntr],nthread=1)
dtr.set_float_info('label_lower_bound',lo[:ntr]); dtr.set_float_info('label_upper_bound',hi[:ntr])
res={}
for ex in sys.argv[4:]:
    exe=os.path.expanduser(ex)
    subprocess.run([exe,'train','/tmp/a_X.csv','/tmp/a_lo.csv','-o','/tmp/a_se.json','--objective','survival:aft','--label-upper-bound','/tmp/a_hi.csv','--aft-distribution',dist,'--aft-scale',str(scale),'--rounds','200','--depth',str(depth),'--eta','0.05'],check=True,capture_output=True)
    res[ex]=nll(xgb.Booster(model_file='/tmp/a_se.json').predict(dte))
p={'objective':'survival:aft','aft_loss_distribution':dist,'aft_loss_distribution_scale':scale,'eta':0.05,'max_depth':depth,'tree_method':'hist','nthread':1}
res['xgb']=nll(xgb.train(p,dtr,200).predict(dte))
for k,v in res.items():
    d=v-res['xgb']; print(k, f'{v.mean():.4f}', f'vs xgb {d.mean():+.4f}±{d.std()/len(d)**.5:.4f}')
