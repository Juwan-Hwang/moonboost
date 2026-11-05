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
  let go = (((w & 0xFFFF) - ranks.get(base + (w >> 16)) - 1).reinterpret_as_uint() >>
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
