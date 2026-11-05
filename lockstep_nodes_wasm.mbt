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
/// One row: several trees walk to their leaves side by side, with no `idx`
/// traffic; a tree shallower than the deepest of its group idles on its
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
  // wasm-gc gains from more loads in flight; plain wasm spills past four
  if WIDE_SINGLE {
    self.single_wide(roots, depth, tree_group, ranks, leaf_value, margin)
  } else {
    self.single_narrow(roots, depth, tree_group, ranks, leaf_value, margin)
  }
}

///|
/// One row, sixteen then four trees side by side, written out without calls.
fn Nodes::single_wide(
  self : Nodes,
  roots : FixedArray[Int32],
  depth : FixedArray[Int32],
  tree_group : Array[Int32],
  ranks : Buf,
  leaf_value : FixedArray[Float],
  margin : FixedArray[Float],
) -> Unit {
  let nodes = self.0
  let trees = roots.length()
  let mut t = 0
  while t + 15 < trees {
    let mut x0 = roots[t + 0]
    let mut x1 = roots[t + 1]
    let mut x2 = roots[t + 2]
    let mut x3 = roots[t + 3]
    let mut x4 = roots[t + 4]
    let mut x5 = roots[t + 5]
    let mut x6 = roots[t + 6]
    let mut x7 = roots[t + 7]
    let mut x8 = roots[t + 8]
    let mut x9 = roots[t + 9]
    let mut x10 = roots[t + 10]
    let mut x11 = roots[t + 11]
    let mut x12 = roots[t + 12]
    let mut x13 = roots[t + 13]
    let mut x14 = roots[t + 14]
    let mut x15 = roots[t + 15]
    let mut d = depth[t]
    for q in 1..<16 {
      if depth[t + q] > d {
        d = depth[t + q]
      }
    }
    for _ in 0..<d {
      x0 = {
        let nd = nodes.unsafe_get(x0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x1 = {
        let nd = nodes.unsafe_get(x1)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x2 = {
        let nd = nodes.unsafe_get(x2)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x3 = {
        let nd = nodes.unsafe_get(x3)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x4 = {
        let nd = nodes.unsafe_get(x4)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x5 = {
        let nd = nodes.unsafe_get(x5)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x6 = {
        let nd = nodes.unsafe_get(x6)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x7 = {
        let nd = nodes.unsafe_get(x7)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x8 = {
        let nd = nodes.unsafe_get(x8)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x9 = {
        let nd = nodes.unsafe_get(x9)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x10 = {
        let nd = nodes.unsafe_get(x10)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x11 = {
        let nd = nodes.unsafe_get(x11)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x12 = {
        let nd = nodes.unsafe_get(x12)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x13 = {
        let nd = nodes.unsafe_get(x13)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x14 = {
        let nd = nodes.unsafe_get(x14)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x15 = {
        let nd = nodes.unsafe_get(x15)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let g = tree_group[t + 0]
    margin[g] = margin[g] + leaf_value[x0]
    let g = tree_group[t + 1]
    margin[g] = margin[g] + leaf_value[x1]
    let g = tree_group[t + 2]
    margin[g] = margin[g] + leaf_value[x2]
    let g = tree_group[t + 3]
    margin[g] = margin[g] + leaf_value[x3]
    let g = tree_group[t + 4]
    margin[g] = margin[g] + leaf_value[x4]
    let g = tree_group[t + 5]
    margin[g] = margin[g] + leaf_value[x5]
    let g = tree_group[t + 6]
    margin[g] = margin[g] + leaf_value[x6]
    let g = tree_group[t + 7]
    margin[g] = margin[g] + leaf_value[x7]
    let g = tree_group[t + 8]
    margin[g] = margin[g] + leaf_value[x8]
    let g = tree_group[t + 9]
    margin[g] = margin[g] + leaf_value[x9]
    let g = tree_group[t + 10]
    margin[g] = margin[g] + leaf_value[x10]
    let g = tree_group[t + 11]
    margin[g] = margin[g] + leaf_value[x11]
    let g = tree_group[t + 12]
    margin[g] = margin[g] + leaf_value[x12]
    let g = tree_group[t + 13]
    margin[g] = margin[g] + leaf_value[x13]
    let g = tree_group[t + 14]
    margin[g] = margin[g] + leaf_value[x14]
    let g = tree_group[t + 15]
    margin[g] = margin[g] + leaf_value[x15]
    t += 16
  }
  while t + 3 < trees {
    let mut x0 = roots[t + 0]
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
      x0 = {
        let nd = nodes.unsafe_get(x0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x1 = {
        let nd = nodes.unsafe_get(x1)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x2 = {
        let nd = nodes.unsafe_get(x2)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x3 = {
        let nd = nodes.unsafe_get(x3)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let g = tree_group[t + 0]
    margin[g] = margin[g] + leaf_value[x0]
    let g = tree_group[t + 1]
    margin[g] = margin[g] + leaf_value[x1]
    let g = tree_group[t + 2]
    margin[g] = margin[g] + leaf_value[x2]
    let g = tree_group[t + 3]
    margin[g] = margin[g] + leaf_value[x3]
    t += 4
  }
  while t + 0 < trees {
    let mut x0 = roots[t + 0]
    let d = depth[t]
    for _ in 0..<d {
      x0 = {
        let nd = nodes.unsafe_get(x0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let g = tree_group[t + 0]
    margin[g] = margin[g] + leaf_value[x0]
    t += 1
  }
}

///|
/// One row, four trees side by side, written out without calls.
fn Nodes::single_narrow(
  self : Nodes,
  roots : FixedArray[Int32],
  depth : FixedArray[Int32],
  tree_group : Array[Int32],
  ranks : Buf,
  leaf_value : FixedArray[Float],
  margin : FixedArray[Float],
) -> Unit {
  let nodes = self.0
  let trees = roots.length()
  let mut t = 0
  while t + 3 < trees {
    let mut x0 = roots[t + 0]
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
      x0 = {
        let nd = nodes.unsafe_get(x0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x1 = {
        let nd = nodes.unsafe_get(x1)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x2 = {
        let nd = nodes.unsafe_get(x2)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      x3 = {
        let nd = nodes.unsafe_get(x3)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let g = tree_group[t + 0]
    margin[g] = margin[g] + leaf_value[x0]
    let g = tree_group[t + 1]
    margin[g] = margin[g] + leaf_value[x1]
    let g = tree_group[t + 2]
    margin[g] = margin[g] + leaf_value[x2]
    let g = tree_group[t + 3]
    margin[g] = margin[g] + leaf_value[x3]
    t += 4
  }
  while t + 0 < trees {
    let mut x0 = roots[t + 0]
    let d = depth[t]
    for _ in 0..<d {
      x0 = {
        let nd = nodes.unsafe_get(x0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(w >> 16)).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let g = tree_group[t + 0]
    margin[g] = margin[g] + leaf_value[x0]
    t += 1
  }
}
