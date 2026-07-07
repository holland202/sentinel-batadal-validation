#!/usr/bin/env python3
"""SENTINEL x BATADAL -- reference validation, single module.
Vincit Omnia Veritas.

Reference implementation of SENTINEL's documented method (two-window
Jensen-Shannon divergence change detection on per-channel histograms)
plus evaluation and per-attack attribution against the public BATADAL
benchmark. NOT the proprietary v12 source; thresholds auto-calibrate
from clean data (99.5th pct) -- no proprietary constants.
Python 3 + numpy only. Data: run fetch_batadal.sh first.
"""
import csv, sys
import numpy as np


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [x for x in r if x]
    idx = [i for i, h in enumerate(hdr) if h not in ("DATETIME", "ATT_FLAG")]
    names = [hdr[i] for i in idx]
    ai = hdr.index("ATT_FLAG")
    X = np.zeros((len(rows), len(idx)))
    y = np.zeros(len(rows), dtype=int)
    for k, row in enumerate(rows):
        for j, i in enumerate(idx):
            try: X[k, j] = float(row[i])
            except ValueError: pass
        try: y[k] = 1 if float(row[ai]) >= 0.5 else 0
        except ValueError: pass
    return names, X, y


def jsd(p, q, eps=1e-12):
    p = p + eps; q = q + eps
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class Detector:
    """Two-window JSD change detector. SEAM: replace fit/score with the
    production v12 detector for authoritative numbers."""

    def __init__(self, window=60, n_bins=16):
        self.W, self.B = int(window), int(n_bins)
        self.edges, self.threshold = None, None

    def fit(self, Xn):
        self.edges = []
        for j in range(Xn.shape[1]):
            lo, hi = Xn[:, j].min(), Xn[:, j].max()
            if hi <= lo: hi = lo + 1.0
            pad = 0.05 * (hi - lo)
            self.edges.append(np.linspace(lo - pad, hi + pad, self.B + 1))
        return self

    def score(self, X, per_channel=False):
        n, c = X.shape
        agg = np.zeros(n)
        chan = np.zeros((n, c)) if per_channel else None
        for t in range(2 * self.W, n):
            cur, ref = X[t - self.W:t], X[t - 2 * self.W:t - self.W]
            v = np.empty(c)
            for j in range(c):
                hc, _ = np.histogram(cur[:, j], bins=self.edges[j])
                hr, _ = np.histogram(ref[:, j], bins=self.edges[j])
                v[j] = jsd(hc.astype(float), hr.astype(float))
            agg[t] = 0.7 * v.max() + 0.3 * v.mean()
            if per_channel: chan[t] = v
        return (agg, chan) if per_channel else agg

    def calibrate(self, clean_scores, pct=99.5):
        w = clean_scores[clean_scores > 0]
        self.threshold = float(np.percentile(w, pct)) if w.size else 0.0
        return self.threshold


def episodes(y):
    out, i = [], 0
    while i < len(y):
        if y[i]:
            j = i
            while j + 1 < len(y) and y[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out


def evaluate(y, s, thr, warm):
    p = (s > thr).astype(int)
    m = np.arange(len(y)) >= warm
    tp = int(((p == 1) & (y == 1) & m).sum()); fp = int(((p == 1) & (y == 0) & m).sum())
    tn = int(((p == 0) & (y == 0) & m).sum()); fn = int(((p == 0) & (y == 1) & m).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    det = sum(1 for a, b in episodes(y) if p[a:b + 1].any())
    return dict(recall=rec, precision=prec, f1=f1,
                fpr=fp / (fp + tn) if fp + tn else 0.0,
                eps=len(episodes(y)), det=det)


# Published BATADAL test-set ground truth (Taormina 2018; K&S 2019 T.VII).
# Attacks 2 and 5 REPLAY L_T2 to SCADA: the attacked channel looks normal,
# so localization via coupled channels is the correct outcome there.
GT = {1: (["L_T3"], ["L_T3", "L_T7", "P_J300", "P_J256"]),
      2: (["L_T2"], ["L_T2", "P_J300", "P_J289", "P_J422"]),
      3: (["F_PU3", "S_PU3"], ["F_PU3", "S_PU3", "L_T1", "F_PU1", "F_PU2", "P_J269"]),
      4: (["F_PU3", "S_PU3"], ["F_PU3", "S_PU3", "L_T1", "F_PU1", "F_PU2", "P_J269"]),
      5: (["L_T2", "F_V2", "S_V2"], ["L_T2", "F_V2", "S_V2", "P_J300", "P_J289", "P_J422"]),
      6: (["L_T7"], ["L_T7", "P_J302", "P_J307"]),
      7: (["L_T4"], ["L_T4", "P_J300", "P_J289", "P_J422"])}


def attribution(names, agg, chan, eps_list, topk=5):
    hits_t = hits_a = 0
    for k, (a, b) in enumerate(eps_list, 1):
        t = a + int(np.argmax(agg[a:b + 1]))
        top = [names[j] for j in np.argsort(chan[t])[::-1][:topk]]
        tgt, area = GT[k]
        ht, ha = any(x in top for x in tgt), any(x in top for x in area)
        hits_t += ht; hits_a += ha
        print(f"  attack {k}: target {tgt[0]:6s} top5 {top} "
              f"-> {'TARGET' if ht else 'AREA' if ha else 'miss'}")
    return hits_t, hits_a


def main(d="./batadal"):
    _, Xtr, ytr = load(f"{d}/dataset03.csv")
    nm, Xt, yt = load(f"{d}/test_dataset.csv")
    print("SENTINEL x BATADAL (live run, reference detector)")
    print(f"clean {Xtr.shape[0]} rows | test {Xt.shape[0]} rows, "
          f"{int(yt.sum())} attack ticks, {len(episodes(yt))} episodes\n")
    print("DETECTION (threshold = 99.5th pct of clean scores)")
    for name, (w, b) in (("Conservative", (100, 12)), ("Balanced", (60, 16)),
                         ("Aggressive", (30, 16))):
        det = Detector(w, b).fit(Xtr)
        thr = det.calibrate(det.score(Xtr))
        e = evaluate(yt, det.score(Xt), thr, 2 * w)
        print(f"  {name:12s} s/bin={w/b:5.2f} episodes {e['det']}/{e['eps']} "
              f"recall {e['recall']*100:5.1f}% precision {e['precision']*100:5.1f}% "
              f"F1 {e['f1']:.3f}")
    print("\nATTRIBUTION vs published ground truth")
    for lbl, (w, b) in (("point w=30 b=12", (30, 12)), ("area  w=48 b=16", (48, 16))):
        det = Detector(w, b).fit(Xtr)
        agg, chan = det.score(Xt, per_channel=True)
        print(f" config {lbl}:")
        ht, ha = attribution(nm, agg, chan, episodes(yt))
        print(f"  => exact-target {ht}/7 | attack-area {ha}/7\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./batadal")
