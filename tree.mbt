///|
/// A single regression tree stored in the flat array layout used by XGBoost.
///
/// Node `i` is a leaf when `left[i] == -1`. For a leaf, `value[i]` is the leaf
/// weight. For a split node, `value[i]` is the split threshold and the sample
/// goes left when `x[feature[i]] < value[i]`. Missing values (NaN) follow
/// `default_left[i]`.
///
/// A node may instead be a categorical split: `categories[i]` holds the sorted
/// category codes that go right; every other non-missing code goes left.
///
/// All arithmetic is done in 32-bit floats, matching XGBoost's `bst_float`,
/// so that comparisons against thresholds behave bit-for-bit the same.
pub struct Tree {
  left : Array[Int32]
  right : Array[Int32]
  feature : Array[Int32]
  value : Array[Float]
  default_left : Array[Bool]
  categories : Array[Array[Int32]?]
}

///|
pub fn Tree::new(
  left~ : Array[Int32],
  right~ : Array[Int32],
  feature~ : Array[Int32],
  value~ : Array[Float],
  default_left~ : Array[Bool],
  categories~ : Array[Array[Int32]?] = [],
) -> Tree raise ModelError {
  let n = left.length()
  if n == 0 {
    raise InvalidModel("tree has no nodes")
  }
  if right.length() != n ||
    feature.length() != n ||
    value.length() != n ||
    default_left.length() != n {
    raise InvalidModel("tree arrays have inconsistent lengths")
  }
  let categories = if categories.length() == 0 {
    Array::make(n, None)
  } else if categories.length() != n {
    raise InvalidModel("categories length \{categories.length()} != \{n}")
  } else {
    categories.map(c => {
      c.map(set => {
        let sorted = set.copy()
        sorted.sort()
        sorted
      })
    })
  }
  for i in 0..<n {
    let l = left[i]
    let r = right[i]
    if l == -1 {
      continue
    }
    if l <= i || r <= i || l >= n || r >= n {
      raise InvalidModel("node \{i} has invalid children (\{l}, \{r})")
    }
    if feature[i] < 0 {
      raise InvalidModel("node \{i} splits on negative feature \{feature[i]}")
    }
  }
  { left, right, feature, value, default_left, categories, }
}

///|
/// Number of nodes in the tree.
pub fn Tree::num_nodes(self : Tree) -> Int32 {
  self.left.length()
}

///|
/// Largest feature index used by any split, or -1 for a single-leaf tree.
pub fn Tree::max_feature(self : Tree) -> Int32 {
  let mut m = -1
  for i in 0..<self.left.length() {
    if self.left[i] != -1 && self.feature[i] > m {
      m = self.feature[i]
    }
  }
  m
}

///|
/// Index of the leaf reached by `row`. Features beyond the end of `row` are
/// treated as missing.
pub fn Tree::leaf_index(self : Tree, row : ArrayView[Float]) -> Int32 {
  let mut node = 0
  while self.left[node] != -1 {
    let f = self.feature[node]
    let missing = f >= row.length() || row[f].is_nan()
    node = if missing {
      if self.default_left[node] {
        self.left[node]
      } else {
        self.right[node]
      }
    } else {
      match self.categories[node] {
        None =>
          if row[f] < self.value[node] {
            self.left[node]
          } else {
            self.right[node]
          }
        Some(set) =>
          if goes_right(set, row[f]) {
            self.right[node]
          } else {
            self.left[node]
          }
      }
    }
  }
  node
}

///|
/// Leaf weight reached by `row`.
pub fn Tree::predict(self : Tree, row : ArrayView[Float]) -> Float {
  self.value[self.leaf_index(row)]
}

///|
/// XGBoost's categorical decision: the value is truncated to an integer code;
/// codes in `set` go right, everything else (including negative and
/// out-of-range codes) goes left.
fn goes_right(set : Array[Int32], x : Float) -> Bool {
  if x < 0.0 || x >= 2147483648.0 {
    return false
  }
  let code = x.to_int()
  set.binary_search(code) is Ok(_)
}
