# Research prototype of moonboost's trainer (numba): histogram GBDT, leaf-wise growth,
# native categorical splits. New ideas are tried here first, then ported to MoonBit.
import numpy as np, numba as nb, time, os, sys, json
from sklearn.metrics import roc_auc_score

# ---------- binning ----------
def make_bins(Xtr, cat, max_bin=255):
    """returns per-feature (kind, cuts or category map). bin 0 = missing/unseen."""
    spec = []
    for f in range(Xtr.shape[1]):
        col = Xtr[:, f]; col = col[~np.isnan(col)]
        if f in cat:
            vals = np.unique(col.astype(np.int64))
            spec.append(('c', vals))
        else:
            u = np.unique(col)
            if len(u) <= max_bin: cuts = u[1:]
            else:
                q = np.quantile(col, np.arange(1, max_bin) / max_bin, method='lower')
                cuts = np.unique(q[q > u[0]])
            spec.append(('n', cuts))
    return spec

def apply_bins(X, spec):
    n, F = X.shape; B = np.zeros((F, n), np.int32)
    for f, (k, s) in enumerate(spec):
        col = X[:, f]; nan = np.isnan(col)
        if k == 'n':
            B[f] = np.searchsorted(s, col, side='right') + 1
        else:
            idx = np.searchsorted(s, np.nan_to_num(col).astype(np.int64))
            idx = np.minimum(idx, len(s) - 1)
            ok = s[idx] == np.nan_to_num(col).astype(np.int64)
            B[f] = np.where(ok, idx + 1, 0)
        B[f][nan] = 0
    return B

# ---------- core ----------
@nb.njit(cache=True)
def hist_build(B, rows, lo, hi, g, h, start, H):
    F = B.shape[0]
    H[:] = 0.0
    for i in range(lo, hi):
        r = rows[i]; gr = g[r]; hr = h[r]
        for f in range(F):
            s = start[f] + B[f, r]
            H[s, 0] += gr; H[s, 1] += hr; H[s, 2] += 1.0

@nb.njit(cache=True)
def best_split(H, start, nb_, iscat, G, Hs, N, lam, mcw, min_leaf, cat_smooth, cat_l2, max_cat, min_cat_group, out_left):
    # returns gain, feature, threshold-bin (numeric) or -1 (categorical, out_left filled), default_left
    parent = G * G / (Hs + lam)
    best = 0.0; bf = -1; bt = -1; bdl = False
    F = start.shape[0]
    bestcat = np.zeros(out_left.shape[0], np.bool_)
    for f in range(F):
        s0 = start[f]; n = nb_[f]
        mg = H[s0, 0]; mh = H[s0, 1]; mc = H[s0, 2]
        if not iscat[f]:
            gl = 0.0; hl = 0.0; cl = 0.0
            for b in range(1, n - 1):
                gl += H[s0 + b, 0]; hl += H[s0 + b, 1]; cl += H[s0 + b, 2]
                if H[s0 + b, 2] == 0: continue
                for dl in range(2):
                    if dl == 1 and mc == 0: continue
                    GL = gl + mg * dl; HL = hl + mh * dl; CL = cl + mc * dl
                    GR = G - GL; HR = Hs - HL; CR = N - CL
                    if HL < mcw or HR < mcw or CL < min_leaf or CR < min_leaf: continue
                    gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
                    if gain > best:
                        best = gain; bf = f; bt = b; bdl = dl == 1
        else:
            # categories with enough data, sorted by g/(h+cat_smooth); missing/unseen bin 0 stays right
            m = 0
            ids = np.empty(n, np.int64); key = np.empty(n)
            for b in range(1, n):
                if H[s0 + b, 2] >= min_cat_group:
                    ids[m] = b; key[m] = H[s0 + b, 0] / (H[s0 + b, 1] + cat_smooth); m += 1
            if m < 2: continue
            order = np.argsort(key[:m]); ids2 = ids[:m][order]
            for direction in range(2):
                gl = 0.0; hl = 0.0; cl = 0.0
                lim = min(m - 1, max_cat)
                for j in range(lim):
                    b = ids2[j] if direction == 0 else ids2[m - 1 - j]
                    gl += H[s0 + b, 0]; hl += H[s0 + b, 1]; cl += H[s0 + b, 2]
                    GR = G - gl; HR = Hs - hl; CR = N - cl
                    if hl < mcw or HR < mcw or cl < min_leaf or CR < min_leaf: continue
                    gain = gl * gl / (hl + lam + cat_l2) + GR * GR / (HR + lam + cat_l2) - parent
                    if gain > best:
                        best = gain; bf = f; bt = -1; bdl = False
                        bestcat[:] = False
                        for jj in range(j + 1):
                            bb = ids2[jj] if direction == 0 else ids2[m - 1 - jj]
                            bestcat[bb] = True
    if bf >= 0 and bt == -1:
        out_left[:] = bestcat
    return best, bf, bt, bdl

@nb.njit(cache=True)
def partition(B, rows, lo, hi, f, t, dl, catleft, iscat, tmp):
    a = lo; b = hi - 1
    for i in range(lo, hi):
        r = rows[i]; v = B[f, r]
        if iscat[f]:
            left = catleft[v]
        else:
            left = (v <= t) if v > 0 else dl
        if left:
            tmp[a] = r; a += 1
        else:
            tmp[b] = r; b -= 1
    for i in range(lo, hi): rows[i] = tmp[i]
    # right part was written reversed; order inside a node does not matter
    return a

@nb.njit(cache=True)
def node_sums(rows, lo, hi, g, h):
    G = 0.0; Hh = 0.0
    for i in range(lo, hi):
        r = rows[i]; G += g[r]; Hh += h[r]
    return G, Hh

@nb.njit(cache=True)
def grow_tree(B, g, h, start, nb_, iscat, maxbins, max_leaves, max_depth, lam, mcw, min_leaf, cat_smooth, cat_l2, max_cat, min_cat_group, lr):
    n = B.shape[1]; F = B.shape[0]; S = start[-1]
    rows = np.arange(n).astype(np.int64); tmp = np.empty(n, np.int64)
    maxn = 2 * max_leaves
    feat = -np.ones(maxn, np.int64); thr = np.zeros(maxn, np.int64); dlft = np.zeros(maxn, np.bool_)
    left = -np.ones(maxn, np.int64); right = -np.ones(maxn, np.int64); val = np.zeros(maxn)
    cats = np.zeros((maxn, maxbins), np.bool_)
    lo_ = np.zeros(maxn, np.int64); hi_ = np.zeros(maxn, np.int64); dep = np.zeros(maxn, np.int64)
    Gs = np.zeros(maxn); Hs_ = np.zeros(maxn)
    gain_ = -np.ones(maxn); sf = -np.ones(maxn, np.int64); st = np.zeros(maxn, np.int64); sdl = np.zeros(maxn, np.bool_)
    hists = np.zeros((maxn, S, 3))
    nn = 1; lo_[0] = 0; hi_[0] = n
    G0, H0 = node_sums(rows, 0, n, g, h); Gs[0] = G0; Hs_[0] = H0
    hist_build(B, rows, 0, n, g, h, start, hists[0])
    catbuf = np.zeros(maxbins, np.bool_)
    gn, f_, t_, d_ = best_split(hists[0], start, nb_, iscat, G0, H0, n, lam, mcw, min_leaf, cat_smooth, cat_l2, max_cat, min_cat_group, catbuf)
    gain_[0] = gn; sf[0] = f_; st[0] = t_; sdl[0] = d_; cats[0] = catbuf
    leaves = 1
    while leaves < max_leaves:
        # best leaf to split
        bi = -1; bg = 1e-12
        for i in range(nn):
            if left[i] < 0 and sf[i] >= 0 and gain_[i] > bg and dep[i] < max_depth:
                bg = gain_[i]; bi = i
        if bi < 0: break
        f = sf[bi]; feat[bi] = f; thr[bi] = st[bi]; dlft[bi] = sdl[bi]
        mid = partition(B, rows, lo_[bi], hi_[bi], f, st[bi], sdl[bi], cats[bi], iscat, tmp)
        l = nn; r = nn + 1; nn += 2; left[bi] = l; right[bi] = r
        lo_[l] = lo_[bi]; hi_[l] = mid; lo_[r] = mid; hi_[r] = hi_[bi]
        dep[l] = dep[bi] + 1; dep[r] = dep[bi] + 1
        Gs[l], Hs_[l] = node_sums(rows, lo_[l], hi_[l], g, h)
        Gs[r] = Gs[bi] - Gs[l]; Hs_[r] = Hs_[bi] - Hs_[l]
        small, big = (l, r) if mid - lo_[bi] <= hi_[bi] - mid else (r, l)
        hist_build(B, rows, lo_[small], hi_[small], g, h, start, hists[small])
        hists[big] = hists[bi] - hists[small]
        leaves += 1
        for c in (l, r):
            if hi_[c] - lo_[c] >= 2 * min_leaf and dep[c] < max_depth:
                gn, f_, t_, d_ = best_split(hists[c], start, nb_, iscat, Gs[c], Hs_[c], hi_[c] - lo_[c], lam, mcw, min_leaf, cat_smooth, cat_l2, max_cat, min_cat_group, catbuf)
                gain_[c] = gn; sf[c] = f_; st[c] = t_; sdl[c] = d_
                if f_ >= 0 and t_ == -1: cats[c] = catbuf
    for i in range(nn):
        if left[i] < 0:
            val[i] = -Gs[i] / (Hs_[i] + lam) * lr
    return nn, feat[:nn], thr[:nn], dlft[:nn], left[:nn], right[:nn], val[:nn], cats[:nn], rows, lo_[:nn], hi_[:nn]

@nb.njit(cache=True)
def predict_tree(B, feat, thr, dlft, left, right, val, cats, iscat, out):
    n = B.shape[1]
    for r in range(n):
        i = 0
        while left[i] >= 0:
            f = feat[i]; v = B[f, r]
            if iscat[f]: go = cats[i, v]
            else: go = (v <= thr[i]) if v > 0 else dlft[i]
            i = left[i] if go else right[i]
        out[r] += val[i]

@nb.njit(cache=True)
def leaf_newton(rows, lo, hi, left, val, m, y, lam, lr, steps):
    # val holds lr * v0; refine v by Newton on the logistic loss of the leaf's rows
    for i in range(left.shape[0]):
        if left[i] >= 0: continue
        v = val[i] / lr
        for s in range(steps):
            G = lam * v; H = lam
            for k in range(lo[i], hi[i]):
                r = rows[k]; p = 1 / (1 + np.exp(-(m[r] + v)))
                G += p - y[r]; H += p * (1 - p)
            v -= G / H
        val[i] = v * lr

def train(Btr, ytr, spec, rounds=100, lr=0.1, max_leaves=1024, max_depth=64, lam=1.0, mcw=1e-3, min_leaf=20,
          cat_smooth=10.0, cat_l2=10.0, max_cat=32, min_cat_group=100, evals=(), hook=None, leaf_steps=0):
    iscat = np.array([k == 'c' for k, _ in spec]); nb_ = np.array([len(s) + 2 if k == 'n' else len(s) + 1 for k, s in spec])
    start = np.concatenate([[0], np.cumsum(nb_)]).astype(np.int64); maxbins = int(nb_.max())
    p0 = ytr.mean(); base = np.log(p0 / (1 - p0))
    m = np.full(len(ytr), base); em = [np.full(B.shape[1], base) for B, _ in evals]
    trees = []
    for it in range(rounds):
        p = 1 / (1 + np.exp(-m)); g = p - ytr; h = p * (1 - p)
        if hook: g, h = hook(it, m, g, h)
        t = grow_tree(Btr, g, h, start, nb_, iscat, maxbins, max_leaves, max_depth, lam, mcw, min_leaf, cat_smooth, cat_l2, max_cat, min_cat_group, lr)
        if leaf_steps: leaf_newton(t[8], t[9], t[10], t[4], t[6], m, ytr, lam, lr, leaf_steps)
        trees.append(t)
        nn, feat, thr, dlft, left, right, val, cats = t[:8]
        predict_tree(Btr, feat, thr, dlft, left, right, val, cats, iscat, m)
        for (B, _), e in zip(evals, em): predict_tree(B, feat, thr, dlft, left, right, val, cats, iscat, e)
    return trees, m, em, iscat

if __name__ == '__main__':
    Xtr = np.load('/tmp/Xtr.npy'); Xte = np.load('/tmp/Xte.npy'); ytr = np.load('/tmp/ytr.npy').astype(np.float64); yte = np.load('/tmp/yte.npy')
    os.sched_setaffinity(0, {int(os.environ.get('CORE', 0))})
    hr = lambda X: np.c_[X, X[:, 3] // 100, X[:, 3] % 100]
    CAT = [0, 1, 2, 5, 6, 7]
    A, B = hr(Xtr), hr(Xte)
    spec = make_bins(A, CAT); Btr = apply_bins(A, spec); Bte = apply_bins(B, spec)
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    t = time.perf_counter()
    trees, m, em, _ = train(Btr, ytr, spec, evals=[(Bte, yte)], **kw)
    print(json.dumps(kw), 'time %.1fs AUC %.4f' % (time.perf_counter() - t, roc_auc_score(yte, em[0])), flush=True)
