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
