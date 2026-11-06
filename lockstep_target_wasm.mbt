///|
/// Whether one-row walks take sixteen trees at a time (see `Nodes::single`).
const WIDE_SINGLE : Bool = false

///|
/// Whether a large compact forest takes bigger batch blocks.
const STREAM_BLOCK : Bool = true

///|
/// Whether ranks come from bucket tables (see `Lockstep::bucket_lo`) rather
/// than a binary search.
const USE_BUCKETS : Bool = true
