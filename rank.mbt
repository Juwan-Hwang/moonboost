///|
/// Gradients and hessians (interleaved, two per row) of LambdaRank for
/// `rank:ndcg`, or of RankNet for `rank:pairwise`, on the queries
/// `starts[q] .. starts[q + 1]`. Each pair of documents with different
/// labels, of which the better scored is in the top `truncation` of its
/// query (all when 0), pushes the better labelled one up with the logistic
/// loss of the score difference. For NDCG, the pair is weighted by the change
/// in NDCG of swapping them, divided by 0.01 plus the score gap, and each
/// query's gradients are scaled by log2(1 + s) / s, s the sum of their
/// magnitudes.
fn rank_gradients(
  label : ArrayView[Float],
  margin : FixedArray[Float],
  starts : FixedArray[Int32],
  inv_max : FixedArray[Double],
  ndcg : Bool,
  truncation : Int32,
) -> FixedArray[Double] {
  let n = label.length()
  let out = FixedArray::make(n * 2, 0.0)
  let disc = discounts(starts)
  let mut longest = 0
  for q in 0..<(starts.length() - 1) {
    let cnt = starts[q + 1] - starts[q]
    if cnt > longest {
      longest = cnt
    }
  }
  // per-query scratch, sized for the longest query
  let keys : Array[UInt64] = Array(capacity=longest)
  let order = FixedArray::make(longest, 0)
  let sc = FixedArray::make(longest, 0.0)
  let lb = FixedArray::make(longest, 0.0)
  let gn = FixedArray::make(longest, 0.0)
  let ex = FixedArray::make(longest, 0.0)
  let lam = FixedArray::make(longest, 0.0)
  let hes = FixedArray::make(longest, 0.0)
  let cls = FixedArray::make(longest, 0)
  let labels = FixedArray::make(max_classes, 0.0)
  let other_len = FixedArray::make(max_classes, 0)
  let next = FixedArray::make(max_classes, 0)
  let list = FixedArray::make(max_classes * longest, 0)
  for q in 0..<(starts.length() - 1) {
    let s = starts[q]
    let cnt = starts[q + 1] - s
    let inv = inv_max[q]
    if cnt < 2 || inv == 0.0 {
      continue
    }
    let k = if truncation <= 0 || truncation > cnt { cnt } else { truncation }
    // documents by current score, best first (ties by position): sorted
    // as keys of the score's bits, flipped so larger sorts first, and the
    // position
    keys.clear()
    for i in 0..<cnt {
      let b = margin[s + i].reinterpret_as_uint()
      let up = if (b & 0x80000000U) != 0U { b.lnot() } else { b | 0x80000000U }
      keys.push((up.lnot().to_uint64() << 32) | i.to_uint64())
    }
    keys.sort()
    for i, key in keys {
      order[i] = s + (key & 0xffffffffUL).to_int()
    }
    // the query in that order: scores, labels, gains, exp(score - best)
    // and the gradient and hessian sums of each position
    let best = margin[order[0]].to_double()
    for i in 0..<cnt {
      let v = margin[order[i]].to_double()
      let l = label[order[i]].to_double()
      sc[i] = v
      lb[i] = l
      gn[i] = gain(l)
      let e = @math.exp(v - best)
      ex[i] = if e < 1.0e-300 { 1.0e-300 } else { e }
      lam[i] = 0.0
      hes[i] = 0.0
    }
    // documents by label class (up to max_classes distinct labels), and
    // for each class the positions, in order, of the documents not in it
    let mut classes = 0
    if ndcg {
      for i in 0..<cnt {
        let l = lb[i]
        let mut c = 0
        while c < classes && labels[c] != l {
          c += 1
        }
        if c == classes {
          if classes == max_classes {
            classes = -1
            break
          }
          labels[c] = l
          classes += 1
        }
        cls[i] = c
      }
      if classes > 0 {
        for c in 0..<classes {
          other_len[c] = 0
          next[c] = c * longest
        }
        for j in 0..<cnt {
          let cj = cls[j]
          for c in 0..<classes {
            if c != cj {
              list[c * longest + other_len[c]] = j
              other_len[c] = other_len[c] + 1
            }
          }
        }
      }
    }
    // LightGBM divides each pair's NDCG change by 0.01 plus the score gap
    // once the scores differ
    let c = if best != sc[cnt - 1] { 0.01 } else { 1.0 }
    let mut sum_l = 0.0
    for i in 0..<k {
      let li = lb[i]
      let si = sc[i]
      let ei = ex[i]
      let gi = gn[i]
      let di = disc[i]
      let mut lam_i = 0.0
      let mut hes_i = 0.0
      if ndcg && classes > 0 {
        // only the documents of other labels: a pair of equal labels adds
        // exactly zero, and the others come in the same increasing order,
        // so every sum is the same. With t the model's chance that i is
        // ahead, the better labelled one is pushed up by (1 - its chance)
        // times the weight. Indices are below cnt, within every scratch
        // array and disc
        let lst = cls.unsafe_get(i)
        let base = lst * longest
        let stop = base + other_len.unsafe_get(lst)
        let mut p = next.unsafe_get(lst)
        while p < stop && list.unsafe_get(p) <= i {
          p += 1
        }
        next.unsafe_set(lst, p)
        for pp in p..<stop {
          let j = list.unsafe_get(pp)
          let ej = ex.unsafe_get(j)
          let e = ei + ej
          let w = c + (si - sc.unsafe_get(j)).abs()
          let r = 1.0 / (e * w)
          let t = ei * w * r
          let dg = gi - gn.unsafe_get(j)
          let ad = dg.abs()
          let f = (di - disc.unsafe_get(j)) * inv * e * r
          let g = (0.5 * dg + ad * (0.5 - t)) * f
          let h = t * (1.0 - t) * ad * f
          lam_i -= g
          lam.unsafe_set(j, lam.unsafe_get(j) + g)
          hes_i += h
          hes.unsafe_set(j, hes.unsafe_get(j) + h)
          sum_l += g.abs()
        }
      } else if ndcg {
        // without branches: a pair of equal labels has no gain gap and adds
        // nothing; with t the model's chance that i is ahead, the better
        // labelled one is pushed up by (1 - its chance) times the weight
        // indices are below cnt, within every scratch array and disc
        for j in (i + 1)..<cnt {
          let ej = ex.unsafe_get(j)
          let e = ei + ej
          let w = c + (si - sc.unsafe_get(j)).abs()
          let r = 1.0 / (e * w)
          let t = ei * w * r
          let dg = gi - gn.unsafe_get(j)
          let ad = dg.abs()
          let f = (di - disc.unsafe_get(j)) * inv * e * r
          let g = (0.5 * dg + ad * (0.5 - t)) * f
          let h = t * (1.0 - t) * ad * f
          lam_i -= g
          lam.unsafe_set(j, lam.unsafe_get(j) + g)
          hes_i += h
          hes.unsafe_set(j, hes.unsafe_get(j) + h)
          sum_l += g.abs()
        }
      } else {
        for j in (i + 1)..<cnt {
          let lj = lb[j]
          if li == lj {
            continue
          }
          let ej = ex[j]
          let t = ei / (ei + ej)
          let g = if li > lj { 1.0 - t } else { -t }
          let h = t * (1.0 - t)
          lam_i -= g
          lam[j] = lam[j] + g
          hes_i += h
          hes[j] = hes[j] + h
        }
      }
      lam[i] = lam[i] + lam_i
      hes[i] = hes[i] + hes_i
    }
    for i in 0..<cnt {
      out[order[i] * 2] = lam[i]
      out[order[i] * 2 + 1] = hes[i]
    }
    if ndcg && sum_l > 0.0 {
      sum_l = 2.0 * sum_l
      let f = log2(1.0 + sum_l) / sum_l
      for i in s..<(s + cnt) {
        out[i * 2] = out[i * 2] * f
        out[i * 2 + 1] = out[i * 2 + 1] * f
      }
    }
  }
  out
}

///|
/// Distinct labels a query may have for rank_gradients to skip the pairs
/// of equal labels; queries with more take every pair.
let max_classes : Int32 = 8

///|
fn gain(label : Double) -> Double {
  @math.pow(2.0, label) - 1.0
}

///|
fn log2(x : Double) -> Double {
  @math.ln(x) / @math.ln(2.0)
}

///|
/// 1 / log2(2 + i) for the positions of the longest query.
fn discounts(starts : FixedArray[Int32]) -> FixedArray[Double] {
  let mut most = 0
  for q in 0..<(starts.length() - 1) {
    let c = starts[q + 1] - starts[q]
    if c > most {
      most = c
    }
  }
  FixedArray::makei(most, i => 1.0 / log2(2.0 + i.to_double()))
}

///|
/// For each query, 1 over its best DCG at the truncation (all documents when
/// 0) for `rank:ndcg`, 0 for a query of no relevant document; 1 for
/// `rank:pairwise`.
fn rank_inverse_max_dcg(
  label : ArrayView[Float],
  starts : FixedArray[Int32],
  ndcg : Bool,
  truncation : Int32,
) -> FixedArray[Double] {
  let nq = starts.length() - 1
  let out = FixedArray::make(nq, 1.0)
  if !ndcg {
    return out
  }
  let disc = discounts(starts)
  let labs : Array[Double] = []
  for q in 0..<nq {
    let s = starts[q]
    let cnt = starts[q + 1] - s
    let k = if truncation <= 0 || truncation > cnt { cnt } else { truncation }
    labs.clear()
    for i in 0..<cnt {
      labs.push(label[s + i].to_double())
    }
    labs.sort_by((a, b) => b.compare(a))
    let mut max_dcg = 0.0
    for i in 0..<k {
      max_dcg += gain(labs[i]) * disc[i]
    }
    out[q] = if max_dcg > 0.0 { 1.0 / max_dcg } else { 0.0 }
  }
  out
}
