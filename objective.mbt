///|
/// How raw tree sums (margins) are turned into predictions.
#derive(Debug, Eq)
pub(all) enum Objective {
  /// Identity link: `reg:squarederror`, `reg:absoluteerror`, `reg:pseudohubererror`,
  /// `reg:quantileerror`, `reg:squaredlogerror`, `binary:logitraw`, `rank:*`.
  Identity(String)
  /// Sigmoid link: `binary:logistic`, `reg:logistic`.
  Logistic(String)
  /// Log link: `count:poisson`, `reg:gamma`, `reg:tweedie`, `survival:cox`.
  Exp(String)
  /// `multi:softprob`: per-class probabilities via softmax.
  Softprob(Int32)
  /// `multi:softmax`: index of the most likely class.
  Softmax(Int32)
}

///|
/// Map an XGBoost objective name to its link function.
pub fn Objective::from_name(
  name : String,
  num_class~ : Int32 = 0,
) -> Objective raise ModelError {
  match name {
    "reg:squarederror"
    | "reg:squaredlogerror"
    | "reg:pseudohubererror"
    | "reg:absoluteerror"
    | "reg:quantileerror"
    | "binary:logitraw"
    | "rank:pairwise"
    | "rank:ndcg"
    | "rank:map" => Identity(name)
    "binary:logistic" | "reg:logistic" => Logistic(name)
    "count:poisson" | "reg:gamma" | "reg:tweedie" | "survival:cox" => Exp(name)
    "multi:softprob" => Softprob(num_class)
    "multi:softmax" => Softmax(num_class)
    _ => raise Unsupported("objective \{name}")
  }
}

///|
/// XGBoost's canonical objective name.
pub fn Objective::name(self : Objective) -> String {
  match self {
    Identity(n) | Logistic(n) | Exp(n) => n
    Softprob(_) => "multi:softprob"
    Softmax(_) => "multi:softmax"
  }
}

///|
/// Convert a user-facing `base_score` into margin space, the way XGBoost's
/// `ProbToMargin` does.
pub fn Objective::base_margin(self : Objective, base_score : Float) -> Float {
  match self {
    Logistic(_) => -@math.lnf((1.0 : Float) / base_score - (1.0 : Float))
    Exp(_) => @math.lnf(base_score)
    _ => base_score
  }
}

///|
fn sigmoid(x : Float) -> Float {
  (1.0 : Float) / ((1.0 : Float) + @math.expf(-x))
}

///|
/// Apply the link function to the margins of one row, in place.
/// For `Softmax` the result has a single element: the predicted class.
pub fn Objective::transform(self : Objective, margins : Array[Float]) -> Unit {
  match self {
    Identity(_) => ()
    Logistic(_) =>
      for i in 0..<margins.length() {
        margins[i] = sigmoid(margins[i])
      }
    Exp(_) =>
      for i in 0..<margins.length() {
        margins[i] = @math.expf(margins[i])
      }
    Softprob(_) => {
      let mut wmax = margins[0]
      for m in margins {
        if m > wmax {
          wmax = m
        }
      }
      let mut sum = 0.0
      for i in 0..<margins.length() {
        let e = @math.expf(margins[i] - wmax)
        margins[i] = e
        sum = sum + e.to_double()
      }
      let s = Float::from_double(sum)
      for i in 0..<margins.length() {
        margins[i] = margins[i] / s
      }
    }
    Softmax(_) => {
      let mut best = 0
      for i in 1..<margins.length() {
        if margins[i] > margins[best] {
          best = i
        }
      }
      margins.clear()
      margins.push(Float::from_int(best))
    }
  }
}

///|
pub extend Objective with Eq::{equal, not_equal}

///|
pub extend Objective with @debug.Debug::{to_repr}
