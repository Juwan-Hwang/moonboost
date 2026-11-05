///|
/// Lockstep nodes packed as `(left << 32) | word`, so a step needs a single
/// load. Used where 64-bit integers are native (everything but JS).
priv struct Nodes(FixedArray[Int64])

///|
fn Nodes::make(n : Int32) -> Nodes {
  FixedArray::make(n, 0L)
}

///|
fn Nodes::set(self : Nodes, id : Int32, left : Int32, word : Int32) -> Unit {
  self.0[id] = (left.to_int64() << 32) | word.to_int64()
}

///|
/// Next node after `i`, reading the row's rank slots at `ranks[base:]`.
/// Callers guarantee every index is in bounds.
fn Nodes::step(self : Nodes, i : Int32, ranks : Buf, base : Int32) -> Int32 {
  let node = self.0.unsafe_get(i)
  let w = node.to_int()
  let go = if ranks.get(base + (w >> 16)) >= (w & 0xFFFF) { 1 } else { 0 }
  (node >> 32).to_int() + go
}

///|
/// Walk trees `a` and `b` (both at least two levels deep, `d` the deeper
/// one) through `n` rows and add their leaves to `margin`, tree `a` first.
/// Node ids interleave in `idx` and each row visit takes two levels, which
/// halves the `idx` traffic on targets with cheap 64-bit loads.
fn Nodes::walk_pair(
  self : Nodes,
  a : Int32,
  b : Int32,
  d : Int32,
  ranks : Buf,
  width : Int32,
  n : Int32,
  idx : Buf,
  leaf_value : FixedArray[Float],
  margin : FixedArray[Float],
  groups : Int32,
  ga : Int32,
  gb : Int32,
) -> Unit {
  for r in 0..<n {
    let base = r * width
    idx.set(2 * r, self.step(a, ranks, base))
    idx.set(2 * r + 1, self.step(b, ranks, base))
  }
  let mut lv = 2
  while lv + 1 < d {
    for r in 0..<n {
      let base = r * width
      let x = self.step(idx.get(2 * r), ranks, base)
      let y = self.step(idx.get(2 * r + 1), ranks, base)
      idx.set(2 * r, self.step(x, ranks, base))
      idx.set(2 * r + 1, self.step(y, ranks, base))
    }
    lv += 2
  }
  if lv < d {
    for r in 0..<n {
      let base = r * width
      idx.set(2 * r, self.step(idx.get(2 * r), ranks, base))
      idx.set(2 * r + 1, self.step(idx.get(2 * r + 1), ranks, base))
    }
  }
  for r in 0..<n {
    let base = r * width
    let l1 = self.step(idx.get(2 * r), ranks, base)
    let l2 = self.step(idx.get(2 * r + 1), ranks, base)
    let k = r * groups + ga
    margin.unsafe_set(k, margin.unsafe_get(k) + leaf_value.unsafe_get(l1))
    let k = r * groups + gb
    margin.unsafe_set(k, margin.unsafe_get(k) + leaf_value.unsafe_get(l2))
  }
}

///|
/// Whether to build and walk the implicit-heap copy of the forest.
const USE_HEAP : Bool = false

///|
/// Largest heap, in slots, per node of the compact forest.
const HEAP_FACTOR : Int32 = 32
