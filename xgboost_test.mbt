///|
test "rejects non-gbtree boosters" {
  let json =
    #|{"learner":{"learner_model_param":{"base_score":"[5E-1]","num_class":"0","num_feature":"1","num_target":"1"},
    #|"objective":{"name":"reg:squarederror"},"gradient_booster":{"name":"gblinear","model":{}}}}
  let r = Ok(@moonboost.Model::from_xgboost_json(json)) catch { e => Err(e) }
  inspect(r is Err(Unsupported(_)), content="true")
}
