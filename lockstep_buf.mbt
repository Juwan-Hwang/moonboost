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

///|
/// Leaf storage, read unchecked.
priv struct FBuf(FixedArray[Float])

///|
fn FBuf::new(n : Int32) -> FBuf {
  FixedArray::make(n, 0.0)
}

///|
fn FBuf::get(self : FBuf, i : Int32) -> Float {
  self.0.unsafe_get(i)
}

///|
fn FBuf::set(self : FBuf, i : Int32, v : Float) -> Unit {
  self.0.unsafe_set(i, v)
}

///|
/// 1 when `a >= b`, else 0.
fn ge01(a : Float, b : Float) -> Int32 {
  if a >= b {
    1
  } else {
    0
  }
}

///|
fn FBuf::length(self : FBuf) -> Int32 {
  self.0.length()
}
