#!/usr/bin/env python3
"""
batadal_score.py - score the reference detector by BATADAL's OWN metric.

The competition does not rank on F1. From the official rules (Taormina,
Galelli, Tippenhauer, Ostfeld, Salomons; "Detailed Problem Description and
Rules", 9 Sep 2016, sec. 3.3-3.4):

    TTD_i   = t_detected - t_start   for attack i; if never detected, TTD = dt
    S_TTD   = 1 - (1/na) * sum_i (TTD_i / dt_i)
    S_CM    = (TPR + TNR) / 2                     <- balanced accuracy, not F1
    S       = gamma * S_TTD + (1 - gamma) * S_CM,  gamma = 0.5

S_CM = 0.5 is the naive baseline (always-safe or always-attack).

Published leaderboard for context (Taormina et al. 2018, Table VII):
    Housh & Ohar 0.970 | Abokifa 0.949 | Giacomoni 0.927 | Brentan 0.894
    Chandy 0.802 | Pasha 0.773 | Aghashahi 0.534

    python3 batadal_score.py ./batadal
"""
import sys

import numpy as np

import sentinel_batadal as sb


def ttd_score(y, pred, warm=0):
    """Time-to-detection score. Undetected attacks contribute TTD = dt."""
    eps = sb.episodes(y)
    if not eps:
        return 1.0, []
    ratios, rows = [], []
    for k, (a, b) in enumerate(eps, 1):
        dt = b - a + 1
        window = pred[a:b + 1].copy()
        if warm:                       # ticks inside warm-up cannot alarm
            window[: max(0, warm - a)] = 0
        hit = np.flatnonzero(window)
        ttd = int(hit[0]) if hit.size else dt
        ratios.append(ttd / dt)
        rows.append((k, a, b, dt, ttd, hit.size > 0))
    return 1.0 - float(np.mean(ratios)), rows


def cm_score(y, pred, warm=0):
    m = np.arange(len(y)) >= warm
    tp = int(((pred == 1) & (y == 1) & m).sum())
    fn = int(((pred == 0) & (y == 1) & m).sum())
    tn = int(((pred == 0) & (y == 0) & m).sum())
    fp = int(((pred == 1) & (y == 0) & m).sum())
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return 0.5 * (tpr + tnr), dict(tp=tp, fp=fp, tn=tn, fn=fn, tpr=tpr, tnr=tnr)


def main(d="./batadal", gamma=0.5):
    stats = {}
    _, Xtr, _ = sb.load(f"{d}/dataset03.csv", stats)
    _, Xt, yt = sb.load(f"{d}/test_dataset.csv", stats)
    print(f"data integrity: {stats['rows']} rows, "
          f"{stats['coerced_x']} sensor / {stats['coerced_y']} label coercions")
    print(f"test set: {len(yt)} ticks, {int(yt.sum())} attack ticks, "
          f"{len(sb.episodes(yt))} episodes\n")

    print("BATADAL OFFICIAL SCORE   S = 0.5*S_TTD + 0.5*S_CM")
    print(f"{'config':<14}{'S_TTD':>8}{'TPR':>8}{'TNR':>8}{'S_CM':>8}"
          f"{'S':>8}   {'F1':>7}  detected")
    best = None
    for name, (w, b) in (("Conservative", (100, 12)), ("Balanced", (60, 16)),
                         ("Aggressive", (30, 16))):
        det = sb.Detector(w, b).fit(Xtr)
        thr = det.calibrate(det.score(Xtr))
        s = det.score(Xt)
        warm = 2 * w
        pred = (s > thr).astype(int)
        sttd, rows = ttd_score(yt, pred, warm)
        scm, cm = cm_score(yt, pred, warm)
        S = gamma * sttd + (1 - gamma) * scm
        f1 = sb.evaluate(yt, s, thr, warm)["f1"]
        ndet = sum(1 for r in rows if r[5])
        print(f"{name:<14}{sttd:>8.3f}{cm['tpr']:>8.3f}{cm['tnr']:>8.3f}"
              f"{scm:>8.3f}{S:>8.3f}   {f1:>7.3f}  {ndet}/7")
        if best is None or S > best[1]:
            best = (name, S, rows, cm, w)

    name, S, rows, cm, w = best
    print(f"\nBest config: {name} (S = {S:.3f})")
    print(f"  vs naive always-safe baseline S_CM = 0.500")
    print(f"  vs published leaderboard: Housh & Ohar 0.970 ... Aghashahi 0.534")
    print(f"\nPer-attack time-to-detection ({name}, warm-up {2*w} ticks):")
    print(f"  {'#':>2}{'start':>7}{'end':>7}{'dur(h)':>8}{'TTD(h)':>8}  detected")
    for k, a, b_, dt, ttd, hit in rows:
        print(f"  {k:>2}{a:>7}{b_:>7}{dt:>8}{ttd:>8}  {'yes' if hit else 'NO'}")
    print(f"\nconfusion (warm-up excluded): TP {cm['tp']} FP {cm['fp']} "
          f"TN {cm['tn']} FN {cm['fn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./batadal"))
