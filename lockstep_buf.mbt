///|
/// Scratch and node storage for the lockstep engine. Reads and writes are
/// unchecked: every index the engine uses is in range by construction.
priv struct Buf(FixedArray[Int32])

///|
fn Buf::new(n : Int32) -> Buf {
  FixedArray::make(n, 0)
}

///|
fn Buf::get(self : Buf, i : Int32) -> Int32 {
  self.0.unsafe_get(i)
}

///|
fn Buf::set(self : Buf, i : Int32, v : Int32) -> Unit {
  self.0.unsafe_set(i, v)
}
