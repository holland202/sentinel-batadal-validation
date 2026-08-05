"""
VERA-on-BATADAL v0 — conformal-certified dynamics anomaly detector
===================================================================
Application prototype: transfer quasar-v2's verified uncertainty layer
(F7/F15 conformal certificates) from the Bloch manifold to a real SCADA
dataset. NOT the 3-dim quantum model — the transferable part is the
method: learn dynamics, certify a residual radius on clean data by split
conformal, flag certificate breaches.

Model: x_{t+1} ~ A x_t + b (ridge, standardized sensors, 43 dims).
Nonconformity: per-tick RMS residual over standardized sensors.
Split: dataset03 (clean year) -> 60% fit / 40% calibrate.
Test: test_dataset with ATT_FLAG ground truth.
Floor: sentinel_batadal.py's published best F1 = 0.251 (tick-level).

REGISTERED (before first run):
  P0  ANTI-VACUITY: clean test ticks breach the 90% certificate at a
      rate that differs from attack ticks' rate. If equal, instrument
      is uninformative and NO claim is made.
  P1  Empirical coverage on clean test ticks in [0.88, 0.92] at nominal
      0.90. (Can fail under drift; failure would be informative.)
  P2  Attack-tick exceedance >= 3x clean-tick exceedance at the 90%
      certificate.
  P3  Tick F1 at the 99% certificate, reported vs floor 0.251 either way.
"""
import csv
import numpy as np

def load(path):
    with open(path) as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [row for row in r if row]
    i_flag = hdr.index("ATT_FLAG")
    sens = [i for i, h in enumerate(hdr) if h not in ("DATETIME", "ATT_FLAG")]
    X = np.array([[float(row[i]) for i in sens] for row in rows])
    y = np.array([int(float(row[i_flag])) for row in rows])
    return X, y

Xtr_all, ytr = load("batadal/dataset03.csv")
Xte, yte = load("batadal/test_dataset.csv")
assert ytr.max() == 0, "dataset03 must be clean"
print(f"train {Xtr_all.shape}, test {Xte.shape}, attack ticks {yte.sum()}"
      f" ({100*yte.mean():.1f}%)")

# standardize with TRAIN stats only
mu, sd = Xtr_all.mean(0), Xtr_all.std(0)
sd[sd < 1e-9] = 1.0
def z(X): return (X - mu) / sd
Ztr, Zte = z(Xtr_all), z(Xte)

# 60/40 fit/calibrate split of the clean year (temporal, no shuffle)
n = len(Ztr); cut = int(0.6 * n)
Zf, Zc = Ztr[:cut], Ztr[cut:]

# ridge dynamics: x_{t+1} = A x_t + b
lam = 1.0
Xf, Yf = Zf[:-1], Zf[1:]
Xa = np.hstack([Xf, np.ones((len(Xf), 1))])
W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Yf)

def resid(Z):
    Xa = np.hstack([Z[:-1], np.ones((len(Z) - 1, 1))])
    return np.sqrt(np.mean((Xa @ W - Z[1:]) ** 2, axis=1))

# split-conformal quantiles from CLEAN calibration residuals
rc = resid(Zc)
def qhat(alpha):
    k = int(np.ceil((len(rc) + 1) * (1 - alpha))) - 1
    return np.sort(rc)[min(k, len(rc) - 1)]
q90, q99 = qhat(0.10), qhat(0.01)
print(f"calibration n={len(rc)}  q90={q90:.4f}  q99={q99:.4f}")

# test-time residuals; align flags to the PREDICTED tick (t+1)
rt = resid(Zte)
flag = yte[1:]
clean, att = rt[flag == 0], rt[flag == 1]

exc_clean90 = float((clean > q90).mean())
exc_att90 = float((att > q90).mean())
cov_clean = 1 - exc_clean90
print(f"\n90% certificate: clean breach {100*exc_clean90:.1f}%  "
      f"attack breach {100*exc_att90:.1f}%  clean coverage {cov_clean:.3f}")

pred99 = rt > q99
tp = int((pred99 & (flag == 1)).sum()); fp = int((pred99 & (flag == 0)).sum())
fn = int((~pred99 & (flag == 1)).sum())
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
print(f"99% certificate detector: precision {prec:.3f}  tick recall "
      f"{rec:.3f}  F1 {f1:.3f}")

print("\n" + "=" * 60 + "\nREGISTERED VERDICTS\n" + "=" * 60)
p0 = abs(exc_att90 - exc_clean90) > 0.05
print(f"P0 anti-vacuity (attack vs clean breach rates differ): {p0}  "
      f"({100*exc_att90:.1f}% vs {100*exc_clean90:.1f}%)")
if not p0:
    print("   -> UNINFORMATIVE. No further claim admissible.")
p1 = 0.88 <= cov_clean <= 0.92
print(f"P1 clean-test coverage in [0.88,0.92]: {p1}  ({cov_clean:.3f})"
      + ("" if p1 else "  <- FAILED, drift between clean year and test"))
p2 = exc_att90 >= 3 * max(exc_clean90, 1e-9)
print(f"P2 attack exceedance >= 3x clean: {p2}  "
      f"({exc_att90/max(exc_clean90,1e-9):.1f}x)")
print(f"P3 F1 {f1:.3f} vs published sentinel floor 0.251: "
      f"{'ABOVE' if f1 > 0.251 else 'BELOW'} floor "
      f"({f1-0.251:+.3f})")
