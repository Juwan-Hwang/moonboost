///|
/// Lockstep nodes packed as `(left << 32) | word`, so a step needs a single
/// load. Used on the wasm targets, where the walk is written out without
/// calls: the engines do not reliably inline a call per tree level.
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
  let go = if ranks.get(base + (w >> 16)) > (w & 0xFFFF) { 1 } else { 0 }
  (node >> 32).to_int() + go
}

///|
/// Walk trees `a` and `b` (both at least two levels deep, `d` the deeper
/// one) through `n` rows and add their leaves to `margin`, tree `a` first.
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
  let nodes = self.0
  for r in 0..<n {
    let base = r * width
    let nd1 = nodes.unsafe_get(a)
    let w1 = nd1.to_int()
    let rk1 = ranks.get(base + (w1 >> 16))
    let p1 = (nd1 >> 32).to_int() +
      (((w1 & 0xFFFF) - rk1).reinterpret_as_uint() >> 31).reinterpret_as_int()
    let nd2 = nodes.unsafe_get(b)
    let w2 = nd2.to_int()
    let rk2 = ranks.get(base + (w2 >> 16))
    let p2 = (nd2 >> 32).to_int() +
      (((w2 & 0xFFFF) - rk2).reinterpret_as_uint() >> 31).reinterpret_as_int()
    idx.set(2 * r, p1)
    idx.set(2 * r + 1, p2)
  }
  let mut lv = 2
  while lv + 1 < d {
    for r in 0..<n {
      let base = r * width
      let nd3 = nodes.unsafe_get(idx.get(2 * r))
      let w3 = nd3.to_int()
      let rk3 = ranks.get(base + (w3 >> 16))
      let x = (nd3 >> 32).to_int() +
        (((w3 & 0xFFFF) - rk3).reinterpret_as_uint() >> 31).reinterpret_as_int()
      let nd4 = nodes.unsafe_get(idx.get(2 * r + 1))
      let w4 = nd4.to_int()
      let rk4 = ranks.get(base + (w4 >> 16))
      let y = (nd4 >> 32).to_int() +
        (((w4 & 0xFFFF) - rk4).reinterpret_as_uint() >> 31).reinterpret_as_int()
      let nd5 = nodes.unsafe_get(x)
      let w5 = nd5.to_int()
      let rk5 = ranks.get(base + (w5 >> 16))
      let x2 = (nd5 >> 32).to_int() +
        (((w5 & 0xFFFF) - rk5).reinterpret_as_uint() >> 31).reinterpret_as_int()
      let nd6 = nodes.unsafe_get(y)
      let w6 = nd6.to_int()
      let rk6 = ranks.get(base + (w6 >> 16))
      let y2 = (nd6 >> 32).to_int() +
        (((w6 & 0xFFFF) - rk6).reinterpret_as_uint() >> 31).reinterpret_as_int()
      idx.set(2 * r, x2)
      idx.set(2 * r + 1, y2)
    }
    lv += 2
  }
  if lv < d {
    for r in 0..<n {
      let base = r * width
      let nd7 = nodes.unsafe_get(idx.get(2 * r))
      let w7 = nd7.to_int()
      let rk7 = ranks.get(base + (w7 >> 16))
      let x = (nd7 >> 32).to_int() +
        (((w7 & 0xFFFF) - rk7).reinterpret_as_uint() >> 31).reinterpret_as_int()
      let nd8 = nodes.unsafe_get(idx.get(2 * r + 1))
      let w8 = nd8.to_int()
      let rk8 = ranks.get(base + (w8 >> 16))
      let y = (nd8 >> 32).to_int() +
        (((w8 & 0xFFFF) - rk8).reinterpret_as_uint() >> 31).reinterpret_as_int()
      idx.set(2 * r, x)
      idx.set(2 * r + 1, y)
    }
  }
  for r in 0..<n {
    let base = r * width
    let nd9 = nodes.unsafe_get(idx.get(2 * r))
    let w9 = nd9.to_int()
    let rk9 = ranks.get(base + (w9 >> 16))
    let l1 = (nd9 >> 32).to_int() +
      (((w9 & 0xFFFF) - rk9).reinterpret_as_uint() >> 31).reinterpret_as_int()
    let nd10 = nodes.unsafe_get(idx.get(2 * r + 1))
    let w10 = nd10.to_int()
    let rk10 = ranks.get(base + (w10 >> 16))
    let l2 = (nd10 >> 32).to_int() +
      (((w10 & 0xFFFF) - rk10).reinterpret_as_uint() >> 31).reinterpret_as_int()
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

///|
/// Whether one-row predictions walk four trees depth-first (needs `step` to
/// be inlined, which the wasm engines do not do).
const SINGLE_WALK : Bool = false
