#!/usr/bin/env python3
"""
test_smoke.py - dataset-free self-tests for sentinel_batadal.py

Runs in seconds with no download. Verifies the maths, the segmentation, the
evaluation arithmetic, the loader's data-integrity counter, and -- critically
-- that the detector CAN return null as well as non-null.

    python3 test_smoke.py              # expect 10/10, exit 0
    python3 test_smoke.py --sabotage   # deliberately break one invariant;
                                       # the suite MUST report failure, exit 1

What this does NOT do: it does not verify the five published BATADAL numbers.
Those need the real dataset (fetch_batadal.sh) and are logged in REPRODUCED.md.
A green run here means the machinery is sound, not that the paper numbers hold.
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

import sentinel_batadal as sb

SABOTAGE = "--sabotage" in sys.argv
RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))


# --------------------------------------------------------------- S1-S3  jsd

def s1_jsd_identity():
    h = np.array([3.0, 9.0, 4.0, 0.0, 1.0])
    d = sb.jsd(h, h)
    check("S1 jsd(h,h) == 0 for identical histograms", abs(d) < 1e-9, f"{d:.3e}")


def s2_jsd_symmetry_and_bounds():
    rng = np.random.default_rng(0)
    ok_sym = ok_bound = True
    worst = 0.0
    for _ in range(50):
        p = rng.random(8) * 10
        q = rng.random(8) * 10
        a, b = sb.jsd(p, q), sb.jsd(q, p)
        ok_sym &= abs(a - b) < 1e-12
        ok_bound &= (-1e-12 <= a <= 1.0 + 1e-9)
        worst = max(worst, a)
    check("S2 jsd is symmetric and stays in [0,1] (log2 base)",
          ok_sym and ok_bound, f"max observed {worst:.4f}")


def s3_jsd_discriminates():
    left = np.array([10.0, 10.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 10.0, 10.0])
    overlap = np.array([5.0, 10.0, 5.0, 0.0])
    d_disjoint = sb.jsd(left, right)
    d_overlap = sb.jsd(left, overlap)
    check("S3 disjoint histograms score strictly above overlapping ones",
          d_disjoint > d_overlap > 0, f"disjoint {d_disjoint:.3f} > overlap {d_overlap:.3f}")


# ------------------------------------------------------------ S4  episodes

def s4_episodes():
    cases = [
        ([0, 0, 0], []),
        ([1, 1, 0, 1], [(0, 1), (3, 3)]),
        ([1], [(0, 0)]),
        ([0, 1, 1, 1, 0], [(1, 3)]),
        ([1, 0, 1, 0, 1], [(0, 0), (2, 2), (4, 4)]),
    ]
    bad = [(v, sb.episodes(np.array(v)), e) for v, e in cases
           if sb.episodes(np.array(v)) != e]
    check("S4 episodes() segments runs correctly incl. edges and singletons",
          not bad, f"{len(cases)} cases" if not bad else f"failed: {bad}")


# ------------------------------------------------------------ S5-S6  eval

def s5_evaluate_extremes():
    y = np.array([0, 0, 1, 1, 0, 0])
    perfect = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    silent = np.zeros(6)
    ep = sb.evaluate(y, perfect, 0.5, 0)
    en = sb.evaluate(y, silent, 0.5, 0)
    check("S5 evaluate() gives F1 1.0 on perfect and 0.0 on a silent detector",
          abs(ep["f1"] - 1.0) < 1e-12 and en["f1"] == 0.0 and en["recall"] == 0.0,
          f"perfect F1 {ep['f1']:.3f}, silent F1 {en['f1']:.3f}")


def s6_warmup_mask():
    y = np.array([1, 1, 0, 0, 1, 1])
    s = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    no_warm = sb.evaluate(y, s, 0.5, 0)
    warmed = sb.evaluate(y, s, 0.5, 4)
    # with warm=4 only ticks 4,5 count: 2 tp, 0 fn -> recall still 1.0,
    # but the earlier attack ticks must be excluded from the episode count basis
    check("S6 warm-up mask excludes early ticks from the confusion matrix",
          no_warm["recall"] == 1.0 and warmed["recall"] == 1.0
          and sb.evaluate(np.array([1, 1, 0, 0, 0, 0]),
                          np.array([1.0, 1.0, 0, 0, 0, 0]), 0.5, 4)["recall"] == 0.0,
          "attack fully inside the warm-up window scores recall 0")


# --------------------------------------------- S7  detector anti-vacuity

def s7_detector_can_return_null_and_signal():
    """Anti-vacuity: the detector must be able to return BOTH null and signal.
    Tested at the README's Conservative operating point (window=100, bins=12)."""
    rng = np.random.default_rng(7)
    flat = np.full((400, 3), 5.0)
    det = sb.Detector(window=100, n_bins=12).fit(flat)
    quiet = det.score(flat)

    noisy = rng.normal(5.0, 0.5, size=(600, 3))
    det2 = sb.Detector(window=100, n_bins=12).fit(noisy)
    shifted = noisy.copy()
    shifted[400:, 0] += 25.0
    loud = det2.score(shifted)
    base, post = loud[200:380].max(), loud[410:].max()

    check("S7 ANTI-VACUITY: flat input scores exactly 0, and at the "
          "Conservative operating point a level shift scores >3x the noise "
          "floor (the detector can return BOTH null and signal)",
          quiet.max() < 1e-9 and post > 3.0 * base,
          f"flat {quiet.max():.2e}, floor {base:.4f}, signal {post:.4f}, "
          f"ratio {post/max(base,1e-9):.2f}x")


def s11_undersampled_window_collapses():
    """DOCUMENTED FAILURE MODE, encoded as a test rather than hidden.
    At window=10 the per-window histograms are so under-sampled that the
    detector's own sampling noise equals or exceeds a 50-sigma level shift.
    This is why the README's recall ordering tracks window size."""
    rng = np.random.default_rng(7)
    noisy = rng.normal(5.0, 0.5, size=(600, 3))
    det = sb.Detector(window=10, n_bins=8).fit(noisy)
    shifted = noisy.copy()
    shifted[400:, 0] += 25.0
    s = det.score(shifted)
    base, post = s[20:380].max(), s[410:].max()
    ratio = post / max(base, 1e-9)
    check("S11 KNOWN LIMIT: at window=10 the noise floor swamps a 25-unit "
          "shift (ratio < 1.5). Small windows are not usable; this is "
          "measured, not hidden",
          ratio < 1.5, f"ratio {ratio:.2f}x -- floor {base:.4f} vs signal {post:.4f}")


def s8_calibrate():
    det = sb.Detector()
    scores = np.concatenate([np.zeros(500), np.linspace(0.0, 1.0, 500)])
    thr = det.calibrate(scores, pct=99.5)
    empty = sb.Detector().calibrate(np.zeros(10))
    check("S8 calibrate() takes the percentile of non-zero scores; "
          "empty input yields 0.0 rather than an error",
          0.9 < thr <= 1.0 and empty == 0.0, f"thr {thr:.4f}, empty {empty}")


# ------------------------------------------------- S9  loader integrity

def _csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def s9_loader_counts_coercions():
    hdr = ["DATETIME", "L_T1", "L_T2", "ATT_FLAG"]
    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "clean.csv"
        dirty = Path(td) / "dirty.csv"
        _csv(clean, [["t1", "1.0", "2.0", "0"], ["t2", "1.5", "2.5", "1"]], hdr)
        _csv(dirty, [["t1", "1.0", "n/a", "0"], ["t2", "", "2.5", "N/A"]], hdr)

        try:
            sc, sd = {}, {}
            _, Xc, _ = sb.load(str(clean), sc)
            _, Xd, yd = sb.load(str(dirty), sd)
        except TypeError:
            check("S9 loader reports unparseable cells instead of silently "
                  "zeroing them", False,
                  "load() does not accept a stats dict - patch_loader.py not applied")
            return

        clean_ok = sc.get("coerced_x", 0) == 0 and sc.get("coerced_y", 0) == 0
        dirty_ok = sd.get("coerced_x", 0) == 2 and sd.get("coerced_y", 0) == 1
        zeroed = Xd[0, 1] == 0.0 and yd[1] == 0
        check("S9 loader counts unparseable cells (clean=0, dirty=2 sensors + "
              "1 label) and still coerces to 0.0 as before",
              clean_ok and dirty_ok and zeroed,
              f"clean {sc.get('coerced_x')}/{sc.get('coerced_y')}, "
              f"dirty {sd.get('coerced_x')}/{sd.get('coerced_y')}")


# ------------------------------------------------- S10  attribution

def s10_attribution_finds_the_culprit():
    rng = np.random.default_rng(10)
    names = ["L_T3", "L_T7", "P_J300", "F_PU1", "L_T1", "S_PU3"]
    W = 12
    base = rng.normal(0.0, 1.0, size=(300, len(names)))
    det = sb.Detector(window=W, n_bins=10).fit(base)
    X = base.copy()
    X[200:240, 0] += 40.0                     # L_T3 is unambiguously the culprit
    y = np.zeros(300, dtype=int)
    y[200:240] = 1
    agg, chan = det.score(X, per_channel=True)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ht, ha = sb.attribution(names, agg, chan, sb.episodes(y))
    check("S10 attribution ranks the injected channel into the top-5 "
          "(exact-target hit on a synthetic ground truth)",
          ht == 1 and ha == 1, f"target hits {ht}, area hits {ha}")


# ------------------------------------------------------------------ runner

def run():
    for fn in (s1_jsd_identity, s2_jsd_symmetry_and_bounds, s3_jsd_discriminates,
               s4_episodes, s5_evaluate_extremes, s6_warmup_mask,
               s7_detector_can_return_null_and_signal, s8_calibrate,
               s9_loader_counts_coercions, s10_attribution_finds_the_culprit,
               s11_undersampled_window_collapses):
        try:
            fn()
        except Exception as exc:                      # a crash is a failure
            RESULTS.append((f"{fn.__name__} raised", False, repr(exc)))

    if SABOTAGE:
        # Prove the suite can fail. Corrupt one invariant and re-check S1.
        real = sb.jsd
        sb.jsd = lambda p, q, eps=1e-12: 0.5        # never zero, never varies
        RESULTS.clear()
        s1_jsd_identity()
        s3_jsd_discriminates()
        sb.jsd = real

    width = max(len(r[0]) for r in RESULTS)
    for label, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} self-tests passed")
    if SABOTAGE:
        print("(--sabotage: failures above are the expected result)")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(run())
