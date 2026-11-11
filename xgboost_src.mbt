///|
/// A node of the parsed model file. On most targets it is MoonBit's `Json`;
/// the JS build uses the engine's own `JSON.parse` instead (see
/// `xgboost_src_js.mbt`), which is many times faster on large models.
priv struct Src(Json)

///|
fn Src::parse(text : StringView) -> Src raise ModelError {
  Src(@json.parse(text) catch { e => raise InvalidModel("invalid JSON: \{e}") })
}

///|
fn Src::field(self : Src, key : String) -> Src raise ModelError {
  match self.0 {
    Object(m) =>
      match m.get(key) {
        Some(v) => Src(v)
        None => raise InvalidModel("missing field `\{key}`")
      }
    _ => raise InvalidModel("`\{key}` should be in an object")
  }
}

///|
fn Src::has(self : Src, key : String) -> Bool {
  self.0 is Object(m) && m.contains(key)
}

///|
fn Src::items(self : Src, what : String) -> Array[Src] raise ModelError {
  match self.0 {
    Array(a) => a.map(v => Src(v))
    _ => raise InvalidModel("`\{what}` should be an array")
  }
}

///|
fn Src::string(self : Src, what : String) -> String raise ModelError {
  match self.0 {
    String(s) => s
    _ => raise InvalidModel("`\{what}` should be a string")
  }
}

///|
fn Src::number(self : Src, what : String) -> Double raise ModelError {
  match self.0 {
    Number(d, ..) => d
    _ => raise InvalidModel("`\{what}` should be a number")
  }
}

///|
fn Src::ints(self : Src, what : String) -> Array[Int32] raise ModelError {
  self.items(what).map(v => v.number(what).to_int())
}

///|
fn Src::floats(self : Src, what : String) -> Array[Float] raise ModelError {
  self.items(what).map(v => Float::from_double(v.number(what)))
}

///|
fn Src::from_json(j : Json) -> Src {
  Src(j)
}
