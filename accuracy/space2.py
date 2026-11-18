# Extends tuned.py's shared search space with row subsampling and per-node
# feature sampling, mapped to each library's names.
import numpy as np, tuned as T
_shared, _mb, _lgb, _xgb = T.shared, T.mb_cfg, T.lgb_cfg, T.xgb_cfg
def shared(r):
  c = _shared(r)
  c['ss'] = float(r.uniform(0.6, 1.0)) if r.rand() < 0.5 else 1.0
  c['cn'] = float(r.uniform(0.5, 1.0)) if r.rand() < 0.5 else 1.0
  return c
def mb_cfg(c):
  return _mb(c) + ['--subsample', '%.3f' % c['ss'], '--colsample-bynode', '%.3f' % c['cn']]
def lgb_cfg(c):
  d = _lgb(c); d.update(bagging_fraction=c['ss'], bagging_freq=1 if c['ss'] < 1 else 0, feature_fraction_bynode=c['cn']); return d
def xgb_cfg(c):
  d = _xgb(c); d.update(subsample=c['ss'], colsample_bynode=c['cn']); return d
T.shared = shared; T.CFG = {'moonboost': mb_cfg, 'lightgbm': lgb_cfg, 'xgboost': xgb_cfg}
