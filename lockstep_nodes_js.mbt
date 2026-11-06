///|
/// Lockstep nodes as pairs of 32-bit integers `[left, word]`. JS has no
/// fast 64-bit integers, so the native packing would be slower there.
priv struct Nodes(Buf)

///|
fn Nodes::make(n : Int32) -> Nodes {
  Buf::new(2 * n)
}

///|
fn Nodes::set(self : Nodes, id : Int32, left : Int32, word : Int32) -> Unit {
  self.0.set(2 * id, left)
  self.0.set(2 * id + 1, word)
}

///|
/// Next node after `i`, reading the row's rank slots at `ranks[base:]`.
/// Callers guarantee every index is in bounds.
fn Nodes::step(self : Nodes, i : Int32, ranks : Buf, base : Int32) -> Int32 {
  let w = self.0.get(2 * i + 1)
  let go = (((w & 0xFFFF) - ranks.get(base + (w >> 16))).reinterpret_as_uint() >>
  31).reinterpret_as_int()
  self.0.get(2 * i) + go
}

///|
/// Walk trees `a` and `b` (both at least two levels deep, `d` the deeper
/// one) through `n` rows and add their leaves to `margin`, tree `a` first.
/// On JS one level per pass over split halves of `idx` is fastest.
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
    idx.set(r, self.step(a, ranks, base))
    idx.set(n + r, self.step(b, ranks, base))
  }
  for _ in 2..<d {
    for r in 0..<n {
      let base = r * width
      idx.set(r, self.step(idx.get(r), ranks, base))
      idx.set(n + r, self.step(idx.get(n + r), ranks, base))
    }
  }
  for r in 0..<n {
    let base = r * width
    let l1 = self.step(idx.get(r), ranks, base)
    let l2 = self.step(idx.get(n + r), ranks, base)
    let k = r * groups + ga
    margin.unsafe_set(k, margin.unsafe_get(k) + leaf_value.unsafe_get(l1))
    let k = r * groups + gb
    margin.unsafe_set(k, margin.unsafe_get(k) + leaf_value.unsafe_get(l2))
  }
}

///|
/// Whether to build and walk the implicit-heap copy of the forest.
const USE_HEAP : Bool = true

///|
/// Largest heap, in slots, per node of the compact forest.
const HEAP_FACTOR : Int32 = 32

///|
/// One row: four trees walk to their leaves side by side, with no `idx`
/// traffic; a tree shallower than the deepest of the four idles on its
/// self-looping leaf. Leaves are added in tree order.
fn Nodes::single(
  self : Nodes,
  roots : FixedArray[Int32],
  depth : FixedArray[Int32],
  tree_group : Array[Int32],
  ranks : Buf,
  leaf_value : FixedArray[Float],
  margin : FixedArray[Float],
) -> Unit {
  let nodes = self
  let trees = roots.length()
  // one row: four trees walk to their leaves side by side, with no `idx`
  // traffic; a tree shallower than the deepest of the four idles on its
  // self-looping leaf
  let mut t = 0
  while t + 3 < trees {
    let mut x0 = roots[t]
    let mut x1 = roots[t + 1]
    let mut x2 = roots[t + 2]
    let mut x3 = roots[t + 3]
    let mut d = depth[t]
    for q in 1..<4 {
      if depth[t + q] > d {
        d = depth[t + q]
      }
    }
    for _ in 0..<d {
      x0 = nodes.step(x0, ranks, 0)
      x1 = nodes.step(x1, ranks, 0)
      x2 = nodes.step(x2, ranks, 0)
      x3 = nodes.step(x3, ranks, 0)
    }
    let g = tree_group[t]
    margin[g] = margin[g] + leaf_value[x0]
    let g = tree_group[t + 1]
    margin[g] = margin[g] + leaf_value[x1]
    let g = tree_group[t + 2]
    margin[g] = margin[g] + leaf_value[x2]
    let g = tree_group[t + 3]
    margin[g] = margin[g] + leaf_value[x3]
    t += 4
  }
  while t < trees {
    let mut x = roots[t]
    for _ in 0..<depth[t] {
      x = nodes.step(x, ranks, 0)
    }
    let g = tree_group[t]
    margin[g] = margin[g] + leaf_value[x]
    t += 1
  }
}

///|
/// Whether a large compact forest takes bigger batch blocks.
const STREAM_BLOCK : Bool = false

///|
/// Whether ranks come from bucket tables (see `Lockstep::bucket_lo`) rather
/// than a binary search.
const USE_BUCKETS : Bool = true
