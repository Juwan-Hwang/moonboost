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
  // four rows by two trees walk to their leaves in locals: eight independent
  // chains keep loads in flight with no `idx` round trip per level
  ignore(idx)
  let mut r = 0
  while r + 3 < n {
    let r0 = r * width
    let r1 = r0 + width
    let r2 = r1 + width
    let r3 = r2 + width
    let mut a0 = a
    let mut c0 = b
    let mut a1 = a
    let mut c1 = b
    let mut a2 = a
    let mut c2 = b
    let mut a3 = a
    let mut c3 = b
    for _ in 0..<d {
      a0 = {
        let nd = nodes.unsafe_get(a0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r0 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      c0 = {
        let nd = nodes.unsafe_get(c0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r0 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      a1 = {
        let nd = nodes.unsafe_get(a1)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r1 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      c1 = {
        let nd = nodes.unsafe_get(c1)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r1 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      a2 = {
        let nd = nodes.unsafe_get(a2)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r2 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      c2 = {
        let nd = nodes.unsafe_get(c2)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r2 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      a3 = {
        let nd = nodes.unsafe_get(a3)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r3 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      c3 = {
        let nd = nodes.unsafe_get(c3)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r3 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let q = (r + 0) * groups
    margin.unsafe_set(
      q + ga,
      margin.unsafe_get(q + ga) + leaf_value.unsafe_get(a0),
    )
    margin.unsafe_set(
      q + gb,
      margin.unsafe_get(q + gb) + leaf_value.unsafe_get(c0),
    )
    let q = (r + 1) * groups
    margin.unsafe_set(
      q + ga,
      margin.unsafe_get(q + ga) + leaf_value.unsafe_get(a1),
    )
    margin.unsafe_set(
      q + gb,
      margin.unsafe_get(q + gb) + leaf_value.unsafe_get(c1),
    )
    let q = (r + 2) * groups
    margin.unsafe_set(
      q + ga,
      margin.unsafe_get(q + ga) + leaf_value.unsafe_get(a2),
    )
    margin.unsafe_set(
      q + gb,
      margin.unsafe_get(q + gb) + leaf_value.unsafe_get(c2),
    )
    let q = (r + 3) * groups
    margin.unsafe_set(
      q + ga,
      margin.unsafe_get(q + ga) + leaf_value.unsafe_get(a3),
    )
    margin.unsafe_set(
      q + gb,
      margin.unsafe_get(q + gb) + leaf_value.unsafe_get(c3),
    )
    r += 4
  }
  while r < n {
    let r0 = r * width
    let mut a0 = a
    let mut c0 = b
    for _ in 0..<d {
      a0 = {
        let nd = nodes.unsafe_get(a0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r0 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
      c0 = {
        let nd = nodes.unsafe_get(c0)
        let w = nd.to_int()
        (nd >> 32).to_int() +
        (((w & 0xFFFF) - ranks.get(r0 + (w >> 16))).reinterpret_as_uint() >> 31).reinterpret_as_int()
      }
    }
    let q = r * groups
    margin.unsafe_set(
      q + ga,
      margin.unsafe_get(q + ga) + leaf_value.unsafe_get(a0),
    )
    margin.unsafe_set(
      q + gb,
      margin.unsafe_get(q + gb) + leaf_value.unsafe_get(c0),
    )
    r += 1
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

///|
/// Whether short forests get the packed value heap (`Lockstep::vheap`).
const VALUE_HEAP : Bool = true

///|
/// A `vheap` split: the order key (`float_key`) of the split value above the
/// rank slot, compared as an integer (`key(x) >= key(value)` goes right),
/// which wasm does without a branch.
fn vheap_split(value : Float, slot : Int32) -> Int64 {
  (float_key(value).to_int64() << 16) | slot.to_int64()
}

///|
/// A padded `vheap` node: a key above every float's, which no row reaches.
const VHEAP_PAD : Int64 = 0x7FFFFFFF0000L

///|
/// Predict a block of rows with the value heap (single output group).
fn Lockstep::predict_vheap(
  self : Lockstep,
  data : FixedArray[Float],
  cols : Int32,
  start : Int32,
  n : Int32,
  keys : Buf,
  vals : FBuf,
  margin : FixedArray[Float],
) -> Unit {
  ignore(vals)
  self.keys_block(data, cols, start, n, keys)
  self.accumulate_vheap(keys, n, margin)
}

///|
/// `accumulate_values` over the packed native copy `vheap` (one output
/// group): rows walk every tree with their margins in locals.
fn Lockstep::accumulate_vheap(
  self : Lockstep,
  vals : Buf,
  n : Int32,
  margin : FixedArray[Float],
) -> Unit {
  let width = 2 * self.num_feature
  let vh = self.vheap
  let leaf = self.vheap_leaf
  let d = self.vheap_depth
  let size = self.heap_size
  let trees = self.roots.length()
  let mut r = 0
  while r + 3 < n {
    let r0 = (r + 0) * width
    let r1 = (r + 1) * width
    let r2 = (r + 2) * width
    let r3 = (r + 3) * width
    let mut m0 = margin.unsafe_get(r + 0)
    let mut m1 = margin.unsafe_get(r + 1)
    let mut m2 = margin.unsafe_get(r + 2)
    let mut m3 = margin.unsafe_get(r + 3)
    let mut t = 0
    while t + 3 < trees {
      let b0 = (t + 0) * size
      let b1 = (t + 1) * size
      let b2 = (t + 2) * size
      let b3 = (t + 3) * size
      let mut x0_0 = 1
      let mut x0_1 = 1
      let mut x0_2 = 1
      let mut x0_3 = 1
      let mut x1_0 = 1
      let mut x1_1 = 1
      let mut x1_2 = 1
      let mut x1_3 = 1
      let mut x2_0 = 1
      let mut x2_1 = 1
      let mut x2_2 = 1
      let mut x2_3 = 1
      let mut x3_0 = 1
      let mut x3_1 = 1
      let mut x3_2 = 1
      let mut x3_3 = 1
      for _ in 0..<d {
        x0_0 = {
          let nd = vh.unsafe_get(b0 + x0_0)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_1 = {
          let nd = vh.unsafe_get(b1 + x0_1)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_1 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_2 = {
          let nd = vh.unsafe_get(b2 + x0_2)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_2 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_3 = {
          let nd = vh.unsafe_get(b3 + x0_3)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_3 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x1_0 = {
          let nd = vh.unsafe_get(b0 + x1_0)
          let kv = vals.get(r1 + (nd & 0xFFFFL).to_int()).to_int64()
          (x1_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x1_1 = {
          let nd = vh.unsafe_get(b1 + x1_1)
          let kv = vals.get(r1 + (nd & 0xFFFFL).to_int()).to_int64()
          (x1_1 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x1_2 = {
          let nd = vh.unsafe_get(b2 + x1_2)
          let kv = vals.get(r1 + (nd & 0xFFFFL).to_int()).to_int64()
          (x1_2 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x1_3 = {
          let nd = vh.unsafe_get(b3 + x1_3)
          let kv = vals.get(r1 + (nd & 0xFFFFL).to_int()).to_int64()
          (x1_3 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x2_0 = {
          let nd = vh.unsafe_get(b0 + x2_0)
          let kv = vals.get(r2 + (nd & 0xFFFFL).to_int()).to_int64()
          (x2_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x2_1 = {
          let nd = vh.unsafe_get(b1 + x2_1)
          let kv = vals.get(r2 + (nd & 0xFFFFL).to_int()).to_int64()
          (x2_1 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x2_2 = {
          let nd = vh.unsafe_get(b2 + x2_2)
          let kv = vals.get(r2 + (nd & 0xFFFFL).to_int()).to_int64()
          (x2_2 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x2_3 = {
          let nd = vh.unsafe_get(b3 + x2_3)
          let kv = vals.get(r2 + (nd & 0xFFFFL).to_int()).to_int64()
          (x2_3 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x3_0 = {
          let nd = vh.unsafe_get(b0 + x3_0)
          let kv = vals.get(r3 + (nd & 0xFFFFL).to_int()).to_int64()
          (x3_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x3_1 = {
          let nd = vh.unsafe_get(b1 + x3_1)
          let kv = vals.get(r3 + (nd & 0xFFFFL).to_int()).to_int64()
          (x3_1 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x3_2 = {
          let nd = vh.unsafe_get(b2 + x3_2)
          let kv = vals.get(r3 + (nd & 0xFFFFL).to_int()).to_int64()
          (x3_2 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x3_3 = {
          let nd = vh.unsafe_get(b3 + x3_3)
          let kv = vals.get(r3 + (nd & 0xFFFFL).to_int()).to_int64()
          (x3_3 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
      }
      m0 = m0 +
        leaf.unsafe_get(b0 + x0_0 - size) +
        leaf.unsafe_get(b1 + x0_1 - size) +
        leaf.unsafe_get(b2 + x0_2 - size) +
        leaf.unsafe_get(b3 + x0_3 - size)
      m1 = m1 +
        leaf.unsafe_get(b0 + x1_0 - size) +
        leaf.unsafe_get(b1 + x1_1 - size) +
        leaf.unsafe_get(b2 + x1_2 - size) +
        leaf.unsafe_get(b3 + x1_3 - size)
      m2 = m2 +
        leaf.unsafe_get(b0 + x2_0 - size) +
        leaf.unsafe_get(b1 + x2_1 - size) +
        leaf.unsafe_get(b2 + x2_2 - size) +
        leaf.unsafe_get(b3 + x2_3 - size)
      m3 = m3 +
        leaf.unsafe_get(b0 + x3_0 - size) +
        leaf.unsafe_get(b1 + x3_1 - size) +
        leaf.unsafe_get(b2 + x3_2 - size) +
        leaf.unsafe_get(b3 + x3_3 - size)
      t += 4
    }
    while t < trees {
      let b0 = (t + 0) * size
      let mut x0_0 = 1
      let mut x1_0 = 1
      let mut x2_0 = 1
      let mut x3_0 = 1
      for _ in 0..<d {
        x0_0 = {
          let nd = vh.unsafe_get(b0 + x0_0)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x1_0 = {
          let nd = vh.unsafe_get(b0 + x1_0)
          let kv = vals.get(r1 + (nd & 0xFFFFL).to_int()).to_int64()
          (x1_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x2_0 = {
          let nd = vh.unsafe_get(b0 + x2_0)
          let kv = vals.get(r2 + (nd & 0xFFFFL).to_int()).to_int64()
          (x2_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x3_0 = {
          let nd = vh.unsafe_get(b0 + x3_0)
          let kv = vals.get(r3 + (nd & 0xFFFFL).to_int()).to_int64()
          (x3_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
      }
      m0 = m0 + leaf.unsafe_get(b0 + x0_0 - size)
      m1 = m1 + leaf.unsafe_get(b0 + x1_0 - size)
      m2 = m2 + leaf.unsafe_get(b0 + x2_0 - size)
      m3 = m3 + leaf.unsafe_get(b0 + x3_0 - size)
      t += 1
    }
    margin.unsafe_set(r + 0, m0)
    margin.unsafe_set(r + 1, m1)
    margin.unsafe_set(r + 2, m2)
    margin.unsafe_set(r + 3, m3)
    r += 4
  }
  while r + 0 < n {
    let r0 = (r + 0) * width
    let mut m0 = margin.unsafe_get(r + 0)
    let mut t = 0
    while t + 3 < trees {
      let b0 = (t + 0) * size
      let b1 = (t + 1) * size
      let b2 = (t + 2) * size
      let b3 = (t + 3) * size
      let mut x0_0 = 1
      let mut x0_1 = 1
      let mut x0_2 = 1
      let mut x0_3 = 1
      for _ in 0..<d {
        x0_0 = {
          let nd = vh.unsafe_get(b0 + x0_0)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_1 = {
          let nd = vh.unsafe_get(b1 + x0_1)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_1 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_2 = {
          let nd = vh.unsafe_get(b2 + x0_2)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_2 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
        x0_3 = {
          let nd = vh.unsafe_get(b3 + x0_3)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_3 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
      }
      m0 = m0 +
        leaf.unsafe_get(b0 + x0_0 - size) +
        leaf.unsafe_get(b1 + x0_1 - size) +
        leaf.unsafe_get(b2 + x0_2 - size) +
        leaf.unsafe_get(b3 + x0_3 - size)
      t += 4
    }
    while t < trees {
      let b0 = (t + 0) * size
      let mut x0_0 = 1
      for _ in 0..<d {
        x0_0 = {
          let nd = vh.unsafe_get(b0 + x0_0)
          let kv = vals.get(r0 + (nd & 0xFFFFL).to_int()).to_int64()
          (x0_0 << 1) | (((kv - (nd >> 16)) >> 63) + 1L).to_int()
        }
      }
      m0 = m0 + leaf.unsafe_get(b0 + x0_0 - size)
      t += 1
    }
    margin.unsafe_set(r + 0, m0)
    r += 1
  }
}

///|
/// Row keys for `accumulate_vheap`: slot `2f` holds the order key
/// (`float_key`) of feature `f`, a missing value as the key of -inf; slot
/// `2f + 1` the same with a missing value as +inf.
fn Lockstep::keys_block(
  self : Lockstep,
  data : FixedArray[Float],
  cols : Int32,
  start : Int32,
  n : Int32,
  out : Buf,
) -> Unit {
  let nf = self.num_feature
  let width = 2 * nf
  let lo = float_key(Float::from_double(-1.0 / 0.0))
  let hi = float_key(Float::from_double(1.0 / 0.0))
  let shared = if cols < nf { cols } else { nf }
  for r in 0..<n {
    let o = r * width
    let at = (start + r) * cols
    for f in 0..<shared {
      let x = data[at + f]
      if x.is_nan() {
        out.set(o + 2 * f, lo)
        out.set(o + 2 * f + 1, hi)
      } else {
        let k = float_key(x)
        out.set(o + 2 * f, k)
        out.set(o + 2 * f + 1, k)
      }
    }
    for f in shared..<nf {
      out.set(o + 2 * f, lo)
      out.set(o + 2 * f + 1, hi)
    }
  }
}
