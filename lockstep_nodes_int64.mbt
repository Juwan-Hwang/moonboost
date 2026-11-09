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
  let go = if ranks.get(base + (w >> 16)) > (w & 0xFFFF) { 1 } else { 0 }
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
      x0 = self.step(x0, ranks, 0)
      x1 = self.step(x1, ranks, 0)
      x2 = self.step(x2, ranks, 0)
      x3 = self.step(x3, ranks, 0)
      x4 = self.step(x4, ranks, 0)
      x5 = self.step(x5, ranks, 0)
      x6 = self.step(x6, ranks, 0)
      x7 = self.step(x7, ranks, 0)
      x8 = self.step(x8, ranks, 0)
      x9 = self.step(x9, ranks, 0)
      x10 = self.step(x10, ranks, 0)
      x11 = self.step(x11, ranks, 0)
      x12 = self.step(x12, ranks, 0)
      x13 = self.step(x13, ranks, 0)
      x14 = self.step(x14, ranks, 0)
      x15 = self.step(x15, ranks, 0)
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
      x0 = self.step(x0, ranks, 0)
      x1 = self.step(x1, ranks, 0)
      x2 = self.step(x2, ranks, 0)
      x3 = self.step(x3, ranks, 0)
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
  while t < trees {
    let mut x = roots[t]
    for _ in 0..<depth[t] {
      x = self.step(x, ranks, 0)
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
const USE_BUCKETS : Bool = false

///|
/// Whether short forests get the packed value heap (`Lockstep::vheap`).
const VALUE_HEAP : Bool = true

///|
/// A `vheap` split: the split value's bits above the rank slot, compared as
/// a float (`x >= value` goes right).
fn vheap_split(value : Float, slot : Int32) -> Int64 {
  (value.reinterpret_as_int().to_int64() << 32) | slot.to_int64()
}

///|
/// A padded `vheap` node: a NaN split value, which no row reaches.
const VHEAP_PAD : Int64 = 0x7FC0000000000000L

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
  ignore(keys)
  self.values_block(data, cols, start, n, vals)
  self.accumulate_vheap(vals, n, margin)
}

///|
/// `accumulate_values` over the packed copy `vheap` (one output group): rows walk every tree with their margins in locals.
fn Lockstep::accumulate_vheap(
  self : Lockstep,
  vals : FBuf,
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
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_0 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_1 = {
          let nd = vh.unsafe_get(b1 + x0_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_1 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_2 = {
          let nd = vh.unsafe_get(b2 + x0_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_2 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_3 = {
          let nd = vh.unsafe_get(b3 + x0_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_3 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x1_0 = {
          let nd = vh.unsafe_get(b0 + x1_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_0 << 1) | ge01(vals.get(r1 + nd.to_int()), c)
        }
        x1_1 = {
          let nd = vh.unsafe_get(b1 + x1_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_1 << 1) | ge01(vals.get(r1 + nd.to_int()), c)
        }
        x1_2 = {
          let nd = vh.unsafe_get(b2 + x1_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_2 << 1) | ge01(vals.get(r1 + nd.to_int()), c)
        }
        x1_3 = {
          let nd = vh.unsafe_get(b3 + x1_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_3 << 1) | ge01(vals.get(r1 + nd.to_int()), c)
        }
        x2_0 = {
          let nd = vh.unsafe_get(b0 + x2_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_0 << 1) | ge01(vals.get(r2 + nd.to_int()), c)
        }
        x2_1 = {
          let nd = vh.unsafe_get(b1 + x2_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_1 << 1) | ge01(vals.get(r2 + nd.to_int()), c)
        }
        x2_2 = {
          let nd = vh.unsafe_get(b2 + x2_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_2 << 1) | ge01(vals.get(r2 + nd.to_int()), c)
        }
        x2_3 = {
          let nd = vh.unsafe_get(b3 + x2_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_3 << 1) | ge01(vals.get(r2 + nd.to_int()), c)
        }
        x3_0 = {
          let nd = vh.unsafe_get(b0 + x3_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_0 << 1) | ge01(vals.get(r3 + nd.to_int()), c)
        }
        x3_1 = {
          let nd = vh.unsafe_get(b1 + x3_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_1 << 1) | ge01(vals.get(r3 + nd.to_int()), c)
        }
        x3_2 = {
          let nd = vh.unsafe_get(b2 + x3_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_2 << 1) | ge01(vals.get(r3 + nd.to_int()), c)
        }
        x3_3 = {
          let nd = vh.unsafe_get(b3 + x3_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_3 << 1) | ge01(vals.get(r3 + nd.to_int()), c)
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
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_0 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x1_0 = {
          let nd = vh.unsafe_get(b0 + x1_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x1_0 << 1) | ge01(vals.get(r1 + nd.to_int()), c)
        }
        x2_0 = {
          let nd = vh.unsafe_get(b0 + x2_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x2_0 << 1) | ge01(vals.get(r2 + nd.to_int()), c)
        }
        x3_0 = {
          let nd = vh.unsafe_get(b0 + x3_0)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x3_0 << 1) | ge01(vals.get(r3 + nd.to_int()), c)
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
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_0 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_1 = {
          let nd = vh.unsafe_get(b1 + x0_1)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_1 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_2 = {
          let nd = vh.unsafe_get(b2 + x0_2)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_2 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
        x0_3 = {
          let nd = vh.unsafe_get(b3 + x0_3)
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_3 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
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
          let c = Float::reinterpret_from_int((nd >> 32).to_int())
          (x0_0 << 1) | ge01(vals.get(r0 + nd.to_int()), c)
        }
      }
      m0 = m0 + leaf.unsafe_get(b0 + x0_0 - size)
      t += 1
    }
    margin.unsafe_set(r + 0, m0)
    r += 1
  }
}
