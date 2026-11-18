# space2 plus min_data (already drawn by tuned.shared and given to LightGBM)
# for moonboost's min_data_in_leaf; XGBoost has no such parameter.
import tuned as T, space2
_mb = T.CFG['moonboost']
def mb_cfg(c):
  return _mb(c) + ['--min-data-in-leaf', str(c['min_data'])]
T.CFG['moonboost'] = mb_cfg
