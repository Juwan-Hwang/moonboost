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
  let go = if ranks.get(base + (w >> 16)) >= (w & 0xFFFF) { 1 } else { 0 }
  self.0.get(2 * i) + go
}
