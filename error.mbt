///|
/// Errors raised while loading or validating a model.
#derive(Show, Eq)
pub(all) suberror ModelError {
  InvalidModel(String)
  Unsupported(String)
}
