///|
/// Errors raised while loading or validating a model.
#derive(Debug, Eq)
pub(all) suberror ModelError {
  InvalidModel(String)
  Unsupported(String)
}

///|
pub extend ModelError with Eq::{equal, not_equal}

///|
pub extend ModelError with @debug.Debug::{to_repr}
