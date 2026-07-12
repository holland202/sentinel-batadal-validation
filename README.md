# SENTINEL x BATADAL — Public Benchmark Validation
*Vincit Omnia Veritas — Truth Conquers All.*

Independent validation of SENTINEL's detection method — two-window
Jensen-Shannon divergence change detection on per-channel sensor
histograms — against the public BATADAL water-distribution attack
benchmark (Taormina et al. 2018). Part of Sovereign Evolution:
edge-native critical-infrastructure defense, built and run entirely
on-device (Android/Termux, Python + NumPy, no cloud).

**Not** the proprietary SENTINEL v12: this is a faithful reference
implementation plus evaluation harness. `class Detector` is the seam
where the production detector drops in. Thresholds auto-calibrate from
clean data (99.5th percentile) — no proprietary constants anywhere.

## Detection results (live runs, test set: 2,089 records, 7 attacks)
| Profile | samples/bin | Episodes | Recall | F1 |
|---|---|---|---|---|
| Conservative (w=100,b=12) | 8.33 | 2/7 | 20.9% | 0.251 |
| Balanced (w=60,b=16) | 3.75 | 4/7 | 15.2% | 0.225 |
| Aggressive (w=30,b=16) | 1.88 | 3/7 | 1.2% | 0.023 |

The histogram under-sampling failure mode is real and reproduced on
external labeled data: at 1.88 samples/bin, recall collapses to ~1%.

## Attribution results (vs published per-attack ground truth)
| Config | Exact target in top-5 | Attack-area in top-5 |
|---|---|---|
| Point (w=30,b=12) | 4/7 | 5/7 |
| Area (w=48,b=16) | 3/7 | **7/7** |

Two honest operating points: the wide window localizes every attack to
a physically-relevant channel; the tight window pins the exact sensor
more often. Attacks 9/12 replay L_T2 to SCADA, so coupled-channel
localization is the correct outcome there. Caveats: reference detector
(a baseline the tuned v12 should beat); exact-sensor localization at
hourly granularity is inherently limited — published literature reports
the same constraint.

## Reproduce

    pip install numpy --break-system-packages
    bash fetch_batadal.sh
    python3 sentinel_batadal.py

Data (c) its authors, not redistributed here — cite Taormina et al. 2018,
J. Water Resour. Plann. Manage. 144(8).

## License
Research use with attribution — see LICENSE.md. Commercial licensing:
c.holland.arch@proton.me

**Chad Edward Holland** (@holland202) · Sovereign Evolution · Oklahoma, USA

