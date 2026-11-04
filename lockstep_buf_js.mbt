///|
/// Scratch and node storage for the lockstep engine: an `Int32Array`, which
/// V8 reads without boxing or hole checks.
#external
priv type Buf

///|
extern "js" fn Buf::new(n : Int32) -> Buf = "(n) => new Int32Array(n)"

///|
extern "js" fn Buf::get(self : Buf, i : Int32) -> Int32 = "(a, i) => a[i]"

///|
extern "js" fn Buf::set(self : Buf, i : Int32, v : Int32) -> Unit = "(a, i, v) => { a[i] = v; }"
