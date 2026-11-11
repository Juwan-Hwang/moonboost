///|
/// A node of the parsed model file, as the JS engine's `JSON.parse` returns
/// it. Arrays of numbers are copied out in one call each.
#external
priv type Src

///|
extern "js" fn js_parse(text : String) -> Src = "(s) => { try { return JSON.parse(s); } catch (e) { return undefined; } }"

///|
extern "js" fn js_defined(v : Src) -> Bool = "(v) => v !== undefined"

///|
extern "js" fn js_get(o : Src, k : String) -> Src = "(o, k) => (o !== null && typeof o === 'object' && !Array.isArray(o) && Object.prototype.hasOwnProperty.call(o, k)) ? o[k] : undefined"

///|
extern "js" fn js_is_array(v : Src) -> Bool = "(v) => Array.isArray(v)"

///|
extern "js" fn js_is_string(v : Src) -> Bool = "(v) => typeof v === 'string'"

///|
extern "js" fn js_items(v : Src) -> FixedArray[Src] = "(v) => v"

///|
extern "js" fn js_string(v : Src) -> String = "(v) => v"

///|
/// The array's numbers truncated toward zero, or `undefined` if any entry
/// is not a number.
extern "js" fn js_ints(v : Src) -> Src = "(v) => { if (!Array.isArray(v)) return undefined; const r = new Array(v.length); for (let i = 0; i < v.length; i++) { const x = v[i]; if (typeof x !== 'number') return undefined; r[i] = Math.trunc(x) | 0; } return r; }"

///|
/// The array's numbers rounded to float32, or `undefined`.
extern "js" fn js_floats(v : Src) -> Src = "(v) => { if (!Array.isArray(v)) return undefined; const r = new Array(v.length); for (let i = 0; i < v.length; i++) { const x = v[i]; if (typeof x !== 'number') return undefined; r[i] = Math.fround(x); } return r; }"

///|
extern "js" fn js_int_array(v : Src) -> FixedArray[Int32] = "(v) => v"

///|
extern "js" fn js_float_array(v : Src) -> FixedArray[Float] = "(v) => v"

///|
fn Src::parse(text : StringView) -> Src raise ModelError {
  let v = js_parse(text.to_owned())
  if !js_defined(v) {
    // rare path: let the MoonBit parser say where the text is broken
    let _ = @json.parse(text) catch {
      e => raise InvalidModel("invalid JSON: \{e}")
    }
    raise InvalidModel("invalid JSON")
  }
  v
}

///|
fn Src::field(self : Src, key : String) -> Src raise ModelError {
  let v = js_get(self, key)
  if !js_defined(v) {
    raise InvalidModel("missing field `\{key}`")
  }
  v
}

///|
fn Src::has(self : Src, key : String) -> Bool {
  js_defined(js_get(self, key))
}

///|
fn Src::items(self : Src, what : String) -> Array[Src] raise ModelError {
  if !js_is_array(self) {
    raise InvalidModel("`\{what}` should be an array")
  }
  Array::from_fixed_array(js_items(self))
}

///|
fn Src::string(self : Src, what : String) -> String raise ModelError {
  if !js_is_string(self) {
    raise InvalidModel("`\{what}` should be a string")
  }
  js_string(self)
}

///|
fn Src::ints(self : Src, what : String) -> Array[Int32] raise ModelError {
  let r = js_ints(self)
  if !js_defined(r) {
    raise InvalidModel("`\{what}` should be an array of numbers")
  }
  Array::from_fixed_array(js_int_array(r))
}

///|
fn Src::floats(self : Src, what : String) -> Array[Float] raise ModelError {
  let r = js_floats(self)
  if !js_defined(r) {
    raise InvalidModel("`\{what}` should be an array of numbers")
  }
  Array::from_fixed_array(js_float_array(r))
}

///|
/// A UBJSON model reaches the JS loader as text, so it goes through the
/// engine's own parser like a JSON file.
fn Src::from_json(j : Json) -> Src raise ModelError {
  Src::parse(j.stringify())
}
