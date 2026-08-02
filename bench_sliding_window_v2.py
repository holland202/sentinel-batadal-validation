#!/usr/bin/env python3
"""bench_sliding_window_v2.py — measure the sliding-window speedup claim.

Three variants of the two-window JSD detector, scored on the real BATADAL
test set, each gated by EXACT equality (np.array_equal) against the
shipped baseline before any timing is reported:

  V0  baseline        Detector.score from sentinel_batadal.py, unmodified
  V1  incremental     O(1) histogram updates per step, per-channel jsd loop
                      (the sliding-window-only change — the tested claim)
  V2  incr+vectorized incremental histograms AND jsd vectorized across
                      all channels (the combined change)

If any variant's aggregate score differs from the baseline in ANY element,
the run FAILS with exit 1 and prints no speedup for that variant. A ratio
from unequal outputs is not a speedup, it is a different program.

--sabotage perturbs one histogram count in V2 by 1 and must exit 1.

Usage:
  python3 bench_sliding_window_v2.py [data_dir] [--sabotage]

Requires sentinel_batadal.py importable from the same directory and
dataset03.csv / test_dataset.csv in data_dir (default ./batadal).
"""
import sys
import time

import numpy as np

import sentinel_batadal as sb


# ---------------------------------------------------------------- variants

def bin_indices(X, edges_list, B):
    """Per-channel bin index for every row, replicating np.histogram
    semantics: half-open bins, last bin closed, out-of-range EXCLUDED
    (index -1)."""
    n, c = X.shape
    idx = np.empty((n, c), dtype=np.int64)
    for j in range(c):
        e = edges_list[j]
        k = np.searchsorted(e, X[:, j], side="right") - 1
        # last edge is inclusive in np.histogram
        k[X[:, j] == e[-1]] = B - 1
        # out of range -> excluded
        k[(X[:, j] < e[0]) | (X[:, j] > e[-1])] = -1
        idx[:, j] = k
    return idx


def score_incremental(det, X, vectorized, sabotage=False):
    """Same output as det.score(X), maintaining both window histograms
    incrementally. vectorized=False keeps the per-channel jsd loop (V1);
    True computes jsd across all channels at once (V2)."""
    n, c = X.shape
    W, B = det.W, det.B
    idx = bin_indices(X, det.edges, B)
    agg = np.zeros(n)
    if n < 2 * W + 1:
        return agg

    def counts(lo, hi):
        h = np.zeros((c, B), dtype=np.int64)
        for j in range(c):
            k = idx[lo:hi, j]
            k = k[k >= 0]
            np.add.at(h[j], k, 1)
        return h

    t0 = 2 * W
    Hc = counts(t0 - W, t0)          # cur  = X[t-W:t]
    Hr = counts(t0 - 2 * W, t0 - W)  # ref  = X[t-2W:t-W]
    if sabotage:
        Hc[0, 0] += 1

    eps = 1e-12
    cols = np.arange(c)

    def jsd_vec(P, Q):
        P = P + eps; Q = Q + eps
        P = P / P.sum(axis=1, keepdims=True)
        Q = Q / Q.sum(axis=1, keepdims=True)
        M = 0.5 * (P + Q)
        return 0.5 * np.sum(P * np.log2(P / M), axis=1) + \
               0.5 * np.sum(Q * np.log2(Q / M), axis=1)

    for t in range(t0, n):
        if t > t0:
            # cur gains row t-1, loses row t-1-W
            a, b = idx[t - 1], idx[t - 1 - W]
            m = a >= 0
            Hc[cols[m], a[m]] += 1
            m = b >= 0
            Hc[cols[m], b[m]] -= 1
            # ref gains row t-1-W, loses row t-1-2W
            m = b >= 0
            Hr[cols[m], b[m]] += 1
            d = idx[t - 1 - 2 * W]
            m = d >= 0
            Hr[cols[m], d[m]] -= 1
        if vectorized:
            v = jsd_vec(Hc.astype(float), Hr.astype(float))
        else:
            v = np.empty(c)
            for j in range(c):
                v[j] = sb.jsd(Hc[j].astype(float), Hr[j].astype(float))
        agg[t] = 0.7 * v.max() + 0.3 * v.mean()
    return agg


# ---------------------------------------------------------------- harness

def main():
    args = [a for a in sys.argv[1:] if a != "--sabotage"]
    sabotage = "--sabotage" in sys.argv[1:]
    d = args[0] if args else "./batadal"

    _, Xtr, _ = sb.load(f"{d}/dataset03.csv", {})
    _, Xt, _ = sb.load(f"{d}/test_dataset.csv", {})
    print(f"platform: python {sys.version.split()[0]}, "
          f"numpy {np.__version__}")
    print(f"data: clean {Xtr.shape} test {Xt.shape}\n")

    failures = []
    for name, (w, b) in (("Conservative", (100, 12)),
                         ("Balanced", (60, 16)),
                         ("Aggressive", (30, 16))):
        det = sb.Detector(w, b).fit(Xtr)

        t = time.perf_counter()
        base = det.score(Xt)
        t_base = time.perf_counter() - t

        t = time.perf_counter()
        v1 = score_incremental(det, Xt, vectorized=False,
                               sabotage=sabotage)
        t_v1 = time.perf_counter() - t

        t = time.perf_counter()
        v2 = score_incremental(det, Xt, vectorized=True,
                               sabotage=sabotage)
        t_v2 = time.perf_counter() - t

        eq1 = np.array_equal(base, v1)
        eq2 = np.array_equal(base, v2)
        print(f"{name:12s} W={w:3d} B={b:2d}  baseline {t_base:7.3f}s")
        if eq1:
            print(f"             V1 incremental      {t_v1:7.3f}s  "
                  f"{t_base / t_v1:6.2f}x  outputs EQUAL")
        else:
            print(f"             V1 incremental      OUTPUTS DIFFER — "
                  f"no speedup reported")
            failures.append(f"{name}-V1")
        if eq2:
            print(f"             V2 incr+vectorized  {t_v2:7.3f}s  "
                  f"{t_base / t_v2:6.2f}x  outputs EQUAL")
        else:
            print(f"             V2 incr+vectorized  OUTPUTS DIFFER — "
                  f"no speedup reported")
            failures.append(f"{name}-V2")
        print()

    if failures:
        print(f"BENCH FAILED — unequal outputs: {failures}")
        sys.exit(1)
    print("all variants exactly equal to baseline; ratios above are valid")
    sys.exit(0)


if __name__ == "__main__":
    main()
