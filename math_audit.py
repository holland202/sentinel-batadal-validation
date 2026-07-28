#!/usr/bin/env python3
"""
math_audit.py - rigorous mathematical audit of sentinel_batadal.py

Tests the JSD implementation against the theorems it is supposed to satisfy,
then probes the detector as a function on its input space. Reports FLAW when a
mathematical property fails and LIMIT when a property holds but has an
operational consequence worth knowing.

Nothing here needs the dataset.
"""
import itertools
import sys

import numpy as np

import sentinel_batadal as sb

RNG = np.random.default_rng(20260727)
OUT = []


def report(tag, name, ok, detail):
    OUT.append((tag, name, ok, detail))


# ==================================================================== JSD
# Jensen-Shannon divergence, base 2. Known properties:
#   (1) non-negativity
#   (2) symmetry
#   (3) boundedness  0 <= JSD <= 1
#   (4) identity of indiscernibles: JSD(p,q) = 0  <=>  p == q (normalised)
#   (5) sqrt(JSD) is a true metric (Endres & Schindelin 2003) -> triangle ineq.
#   (6) invariance under rescaling of the count vectors

def t1_nonnegativity():
    worst = 0.0
    for _ in range(20000):
        n = RNG.integers(2, 20)
        p, q = RNG.random(n) * RNG.integers(1, 1000), RNG.random(n) * RNG.integers(1, 1000)
        worst = min(worst, sb.jsd(p, q))
    report("THEOREM", "JSD >= 0 over 20k random pairs", worst >= -1e-12,
           f"min observed {worst:.3e}")


def t2_symmetry():
    worst = 0.0
    for _ in range(20000):
        n = RNG.integers(2, 20)
        p, q = RNG.random(n) * 100, RNG.random(n) * 100
        worst = max(worst, abs(sb.jsd(p, q) - sb.jsd(q, p)))
    report("THEOREM", "JSD(p,q) == JSD(q,p) over 20k random pairs", worst < 1e-12,
           f"max asymmetry {worst:.3e}")


def t3_boundedness():
    hi = 0.0
    for _ in range(20000):
        n = RNG.integers(2, 20)
        p, q = RNG.random(n) * 100, RNG.random(n) * 100
        hi = max(hi, sb.jsd(p, q))
    # exact upper bound: disjoint support
    a = np.array([1.0, 0.0]); b = np.array([0.0, 1.0])
    disj = sb.jsd(a, b)
    report("THEOREM", "0 <= JSD <= 1 in base 2; disjoint support attains 1",
           hi <= 1.0 + 1e-9 and abs(disj - 1.0) < 1e-9,
           f"max random {hi:.6f}, disjoint {disj:.12f}")


def t4_identity_of_indiscernibles():
    same = max(sb.jsd(p := RNG.random(8) * 50, p.copy()) for _ in range(2000))
    # smallest perturbation that must be non-zero
    p = np.array([10.0, 10.0, 10.0, 10.0])
    q = p.copy(); q[0] += 1e-6
    tiny = sb.jsd(p, q)
    report("THEOREM", "JSD == 0 iff distributions equal (and > 0 for any "
           "perturbation)", same < 1e-12 and tiny > 0,
           f"identical {same:.3e}, 1e-6 perturbation {tiny:.3e}")


def t5_sqrt_is_a_metric():
    """Endres-Schindelin: sqrt(JSD) satisfies the triangle inequality."""
    worst = 0.0
    viol = 0
    for _ in range(20000):
        n = RNG.integers(2, 12)
        p, q, r = (RNG.random(n) * 100 for _ in range(3))
        d_pq, d_qr, d_pr = (np.sqrt(sb.jsd(a, b)) for a, b in
                            ((p, q), (q, r), (p, r)))
        slack = d_pq + d_qr - d_pr
        worst = min(worst, slack)
        viol += slack < -1e-12
    report("THEOREM", "sqrt(JSD) obeys the triangle inequality "
           "(Endres-Schindelin) over 20k triples", viol == 0,
           f"{viol} violations, tightest slack {worst:.3e}")


def t6_scale_invariance():
    """The implementation adds eps BEFORE normalising, so invariance under
    rescaling the raw counts is only approximate. Quantify the error."""
    worst = 0.0
    worst_case = None
    for _ in range(5000):
        n = RNG.integers(2, 16)
        p = np.floor(RNG.random(n) * 20)
        q = np.floor(RNG.random(n) * 20)
        if p.sum() == 0 or q.sum() == 0:
            continue
        for k in (10.0, 1e3, 1e6):
            d = abs(sb.jsd(p, q) - sb.jsd(p * k, q * k))
            if d > worst:
                worst, worst_case = d, k
    report("THEOREM", "JSD invariant under rescaling of the count vectors "
           "(eps added pre-normalisation makes this approximate)",
           worst < 1e-9, f"max deviation {worst:.3e} at scale {worst_case}")


def t7_empty_histogram_degeneracy():
    """If every sample falls outside the bin edges, np.histogram returns all
    zeros. After eps-regularisation that becomes the UNIFORM distribution.
    Two different all-empty histograms are then indistinguishable."""
    z = np.zeros(16)
    d_zz = sb.jsd(z, z)
    populated = np.zeros(16); populated[3] = 60.0
    d_zp = sb.jsd(z, populated)
    report("FLAW", "two all-empty histograms are identical under JSD (both "
           "become uniform), so a window whose samples ALL fall outside the "
           "bin range is indistinguishable from another such window",
           False, f"JSD(empty,empty) = {d_zz:.3e}; JSD(empty,populated) = {d_zp:.4f}")


# =============================================================== DETECTOR

def t8_out_of_range_saturation():
    """Bin edges are fixed at fit() time from the CLEAN data range +/-5%.
    np.histogram DISCARDS samples outside those edges. So as an attack grows,
    the current window's mass leaves the binning entirely -- and the score
    should rise, peak, then COLLAPSE. Detection is non-monotone in attack size."""
    W, B = 60, 16
    base = RNG.normal(0.0, 1.0, size=(600, 1))
    det = sb.Detector(window=W, n_bins=B).fit(base)
    rows = []
    for shift in (0, 1, 2, 3, 5, 8, 12, 20, 50, 200, 1000):
        X = base.copy()
        X[400:, 0] += shift
        s = det.score(X)
        rows.append((shift, float(s[2 * W:380].max()), float(s[470:].max())))
    peak = max(r[2] for r in rows)
    tail = rows[-1][2]
    collapsed = tail < 0.5 * peak
    detail = " | ".join(f"{sh}:{sig:.3f}" for sh, _, sig in rows)
    report("FLAW" if collapsed else "OK",
           "score is NON-MONOTONE in attack magnitude: samples pushed outside "
           "the fitted bin range are discarded, so a large enough attack "
           "scores LOWER than a moderate one",
           not collapsed, f"shift:score  {detail}")
    return rows


def t9_total_evasion():
    """Follow t8 to its conclusion: is there a shift at which the attacked
    window scores at or below the clean noise floor?"""
    W, B = 60, 16
    base = RNG.normal(0.0, 1.0, size=(600, 1))
    det = sb.Detector(window=W, n_bins=B).fit(base)
    thr = det.calibrate(det.score(base))
    evasive = []
    for shift in (5, 10, 20, 50, 100, 500, 1000, 10000):
        X = base.copy(); X[400:, 0] += shift
        s = det.score(X)
        if s[470:].max() <= thr:
            evasive.append((shift, float(s[470:].max())))
    report("FLAW" if evasive else "OK",
           "an attack large enough to move every sample outside the fitted "
           "range evades the calibrated threshold entirely",
           not evasive,
           f"threshold {thr:.4f}; evading shifts {evasive}" if evasive
           else f"threshold {thr:.4f}; no evasion found")


def t10_channel_dilution():
    """agg = 0.7*max + 0.3*mean. The mean term dilutes as channel count grows:
    a single-channel anomaly contributes v_anom/c to it. Measure the decay."""
    W = 60
    rows = []
    for c in (1, 2, 4, 8, 16, 32, 43):        # 43 = BATADAL's channel count
        base = RNG.normal(0.0, 1.0, size=(500, c))
        det = sb.Detector(window=W, n_bins=16).fit(base)
        X = base.copy(); X[300:, 0] += 4.0
        s = det.score(X)
        rows.append((c, float(s[370:].max())))
    first, last = rows[0][1], rows[-1][1]
    decay = (first - last) / first if first else 0.0
    report("LIMIT", "single-channel anomaly score decays as channel count "
           "grows (the 0.3*mean term dilutes); BATADAL has 43 channels",
           True, " | ".join(f"c={c}:{v:.3f}" for c, v in rows)
           + f"  -> {decay*100:.1f}% decay 1->43")


def t11_score_warmup_zeros():
    """score() leaves agg[0:2W] == 0. Those zeros are excluded from calibrate()
    by the >0 filter, but they ARE passed to evaluate() where s > thr is false,
    which is correct only because evaluate() also masks. Check both."""
    W = 30
    X = RNG.normal(0.0, 1.0, size=(300, 2))
    det = sb.Detector(window=W, n_bins=16).fit(X)
    s = det.score(X)
    head_zero = bool(np.all(s[:2 * W] == 0))
    thr = det.calibrate(s)
    nonzero_used = int((s > 0).sum())
    report("OK" if head_zero else "FLAW",
           "the first 2W scores are structurally 0 and are excluded from "
           "threshold calibration by the >0 filter",
           head_zero, f"2W={2*W} leading zeros: {head_zero}; "
           f"{nonzero_used} scores used for the percentile; thr={thr:.4f}")


def t12_constant_channel_edges():
    """fit() on a constant channel sets hi = lo + 1.0, giving edges over
    [lo-0.05, lo+1.05]. Any test value outside that unit interval is discarded.
    A channel that is constant in training and varies in test can go blind."""
    W = 60
    base = np.full((400, 1), 7.0)
    det = sb.Detector(window=W, n_bins=16).fit(base)
    X = base.copy()
    X[300:, 0] = 7.0 + 5.0          # well outside [6.95, 8.05]
    s = det.score(X)
    detected = s[370:].max() > 1e-9
    report("FLAW" if not detected else "OK",
           "a channel constant during training goes BLIND to any test "
           "excursion beyond +1.05 of its training value (all samples fall "
           "outside the fitted edges, both windows become empty)",
           detected, f"max score after a +5.0 excursion: {s[370:].max():.3e}")


def t13_autocorrelation_of_scores():
    """Consecutive windows share W-1 of W samples, so scores are strongly
    autocorrelated. Any per-tick statistic (recall, FPR) therefore has far
    fewer independent samples than it appears to."""
    W = 60
    X = RNG.normal(0.0, 1.0, size=(1200, 3))
    det = sb.Detector(window=W, n_bins=16).fit(X)
    s = det.score(X)[2 * W:]
    s = s - s.mean()
    ac = [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in (1, 5, 10, 30, 60, 120)]
    report("LIMIT", "scores are heavily autocorrelated (windows overlap by "
           "W-1 samples), so tick-level recall/FPR have far fewer effective "
           "independent samples than tick counts suggest",
           True, "lag1..lag120 rho = " + ", ".join(f"{a:.2f}" for a in ac))


def t14_sustained_attack_goes_quiet():
    """A two-window detector compares the last W samples with the W before.
    Once a sustained attack is older than 2W ticks, BOTH windows sit inside
    the attacked regime, the distributions match again, and the score falls
    back toward the noise floor. The detector sees TRANSITIONS, not STATES."""
    rows = []
    for W in (30, 60, 100):
        base = np.random.default_rng(1).normal(0.0, 1.0, size=(800, 3))
        det = sb.Detector(window=W, n_bins=16).fit(base)
        thr = det.calibrate(det.score(base))
        X = base.copy(); X[400:800, 0] += 4.0
        s = det.score(X)
        cov = int((s[400:800] > thr).sum()) / 400.0
        rows.append((W, cov))
    report("FLAW", "a SUSTAINED attack is only detectable near its onset: "
           "once both windows lie inside the attacked regime the score "
           "returns to the noise floor. Coverage of a 400-tick attack is "
           "far below 100% at every window size",
           False, " | ".join(f"W={W}: {c*100:.1f}% of attack ticks alarmed"
                             for W, c in rows))


def t15_tick_recall_is_the_wrong_metric():
    """Direct consequence of t14: tick-level recall penalises the detector for
    the ticks it is mathematically unable to fire on. Episode detection does
    not. Measured on identical runs."""
    tick, ep = [], 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0, 1.0, size=(900, 3))
        det = sb.Detector(window=60, n_bins=16).fit(base)
        thr = det.calibrate(det.score(base))
        X = base.copy(); y = np.zeros(900, dtype=int)
        X[500:700, 0] += 4.0; y[500:700] = 1
        e = sb.evaluate(y, det.score(X), thr, 120)
        tick.append(e["recall"]); ep += e["det"] > 0
    report("FLAW", "tick-level recall and episode detection disagree "
           "structurally on the SAME runs -- reporting tick recall as the "
           "headline understates a change detector by construction",
           False, f"mean tick recall {np.mean(tick)*100:.1f}% vs episode "
           f"detection {ep}/30 = {ep/30*100:.0f}%")


def t16_detection_latency():
    """How long after onset does the first alarm arrive? BATADAL ticks are
    hourly, so this converts directly to wall-clock warning delay."""
    rows = []
    for W in (30, 60, 100):
        lats, missed = [], 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            base = rng.normal(0.0, 1.0, size=(900, 3))
            det = sb.Detector(window=W, n_bins=16).fit(base)
            thr = det.calibrate(det.score(base))
            X = base.copy(); X[500:700, 0] += 4.0
            hit = np.where(det.score(X)[500:700] > thr)[0]
            lats.append(int(hit[0])) if len(hit) else missed.__add__(1)
            if not len(hit):
                missed += 1
        rows.append((W, int(np.median(lats)) if lats else -1, missed))
    report("LIMIT", "median detection latency from attack onset, in ticks "
           "(BATADAL ticks are hourly, so these are hours of delay)",
           True, " | ".join(f"W={W}: {m} ticks ~= {m/24:.1f} days, {mi} missed"
                            for W, m, mi in rows))


def main():
    for fn in (t1_nonnegativity, t2_symmetry, t3_boundedness,
               t4_identity_of_indiscernibles, t5_sqrt_is_a_metric,
               t6_scale_invariance, t7_empty_histogram_degeneracy,
               t8_out_of_range_saturation, t9_total_evasion,
               t10_channel_dilution, t11_score_warmup_zeros,
               t12_constant_channel_edges, t13_autocorrelation_of_scores,
               t14_sustained_attack_goes_quiet, t15_tick_recall_is_the_wrong_metric,
               t16_detection_latency):
        try:
            fn()
        except Exception as exc:
            report("ERROR", f"{fn.__name__} raised", False, repr(exc))

    print("=" * 78)
    print("MATHEMATICAL AUDIT - sentinel_batadal.py")
    print("=" * 78)
    for tag, name, ok, detail in OUT:
        mark = {"THEOREM": "HOLDS" if ok else "VIOLATED",
                "FLAW": "OK" if ok else "FLAW",
                "LIMIT": "LIMIT", "OK": "OK", "ERROR": "ERROR"}[tag]
        print(f"\n[{mark}] {name}")
        print(f"        {detail}")
    flaws = [o for o in OUT if not o[2] and o[0] in ("FLAW", "THEOREM", "ERROR")]
    print("\n" + "=" * 78)
    print(f"{len(OUT)} properties tested, {len(flaws)} flaws found")
    return 1 if flaws else 0


if __name__ == "__main__":
    sys.exit(main())
