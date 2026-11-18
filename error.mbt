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

///|
/// `InvalidModel: <message>` or `Unsupported: <message>`, so a caught error
/// can be printed with `println("\{e}")`.
pub impl Show for ModelError with fn output(self, logger) {
  match self {
    InvalidModel(msg) => {
      logger.write_string("InvalidModel: ")
      logger.write_string(msg)
    }
    Unsupported(msg) => {
      logger.write_string("Unsupported: ")
      logger.write_string(msg)
    }
  }
}
