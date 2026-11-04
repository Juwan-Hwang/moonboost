///|
test "valid forest" {
  // tree 0: root splits feature 1 at key 10; tree 1: single leaf
  let nodes : FixedArray[Int32] = [
    1,
    2,
    1 * 4 + 1,
    10, // node 0, default left
    -1,
    -1,
    0,
    0, // node 1
    -1,
    -1,
    0,
    0, // node 2
    -1,
    -1,
    0,
    0, // node 3 (tree 1)
  ]
  let f = @kernel.Forest::new(nodes, [0, 3], [], []).unwrap()
  inspect(f.num_trees(), content="2")
  inspect(f.num_nodes(), content="4")
  inspect(f.leaf(0, [0, 5]), content="1")
  inspect(f.leaf(0, [0, 10]), content="2")
  inspect(f.leaf(0, [0, @kernel.MISSING]), content="1")
  inspect(f.leaf(0, [0]), content="1") // feature past the end is missing
  inspect(f.leaf(1, [0, 5]), content="3")
  inspect(f.leaf(2, [0, 5]), content="-1")
}

///|
test "categorical split" {
  // codes {2, 7} go right; missing goes right (default_left = 0)
  let nodes : FixedArray[Int32] = [
    1,
    2,
    0 * 4 + 2,
    0,
    -1,
    -1,
    0,
    0,
    -1,
    -1,
    0,
    0,
  ]
  let f = @kernel.Forest::new(nodes, [0], [0, 2], [2, 7]).unwrap()
  inspect(f.leaf(0, [7]), content="2")
  inspect(f.leaf(0, [2]), content="2")
  inspect(f.leaf(0, [3]), content="1")
  inspect(f.leaf(0, [-1]), content="1")
  inspect(f.leaf(0, [@kernel.MISSING]), content="2")
}

///|
test "malformed forests are rejected" {
  let leaf : FixedArray[Int32] = [-1, -1, 0, 0]
  // child pointing backwards (would loop forever)
  inspect(
    @kernel.Forest::new([1, 1, 0, 0, 0, 0, 0, 0], [0], [], []) is None,
    content="true",
  )
  // child out of range
  inspect(
    @kernel.Forest::new([5, 6, 0, 0], [0], [], []) is None,
    content="true",
  )
  // root out of range
  inspect(@kernel.Forest::new(leaf, [1], [], []) is None, content="true")
  // length not a multiple of 4
  inspect(@kernel.Forest::new([-1, -1, 0], [0], [], []) is None, content="true")
  // categorical range outside `cats`
  inspect(
    @kernel.Forest::new([1, 2, 2, 0, -1, -1, 0, 0, -1, -1, 0, 0], [0], [0, 3], [
      1,
    ])
    is None,
    content="true",
  )
  // negative feature
  inspect(
    @kernel.Forest::new([1, 2, -4, 0, -1, -1, 0, 0, -1, -1, 0, 0], [0], [], [])
    is None,
    content="true",
  )
}
