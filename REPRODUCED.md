# Reproduction log

Independent reproductions of `python3 sentinel_batadal.py`. Failures and
corrections are logged here with equal prominence to successes.

---

## 2026-07-27 — CORRECTION: this file previously documented the wrong project

**Every line of this file before this date was about a different repository.**

The published `REPRODUCED.md` was a byte-identical copy of the one in
[qsleuth](https://github.com/holland202/qsleuth). It logged nine self-tests of
a quantum state tomography tool — Bloch spheres, CPTP maps, conformal
coverage — under the heading "ALL 9 SELF-TESTS PASSED. Every README number
reproduced exactly." It referenced a commit and a `qsleuth.py` sha256 that do
not exist here, and contained zero mentions of BATADAL.

A reader would reasonably have concluded that this repo's numbers had been
independently reproduced by nine self-tests. They had not. Until today this
repository had no self-tests at all.

Root cause: the file was copied when the repository was created and never read
in place. Nobody checked that a document named REPRODUCED matched the thing it
claimed to reproduce. It is recorded rather than quietly replaced, because a
repository arguing for verifiable claims does not get to delete its own
counterexample.

---

## 2026-07-27 — Full reproduction on the real data (Claude, Anthropic; sandbox)

Cold run against the official CSVs. **All five published numbers reproduce
exactly.**

```
data integrity: 10850 rows loaded, 0 sensor cells and 0 labels coerced to 0,
                0 rows unlabeled (-999) (clean)
clean 8761 rows | test 2089 rows, 407 attack ticks, 7 episodes

  Conservative  episodes 2/7  recall 20.9%  precision 31.5%  F1 0.251
  Balanced      episodes 4/7  recall 15.2%  precision 42.8%  F1 0.225
  Aggressive    episodes 3/7  recall  1.2%  precision 19.2%  F1 0.023

  attribution point w=30 b=12 : exact-target 4/7 | attack-area 5/7
  attribution area  w=48 b=16 : exact-target 3/7 | attack-area 7/7
```

The `data integrity` line is new and matters: `load()` previously coerced any
unparseable cell to `0.0` in silence. For BATADAL that is a tank level or a
pressure, where zero is physically extreme. The count is now printed, and on
the real data it is **zero** — so the published numbers are not corrupted by
it. That was previously an assumption; it is now a measurement.

### The official BATADAL score

The competition does not rank on F1. Per the rules (Taormina et al., *Detailed
Problem Description and Rules*, 9 Sep 2016, §3.3–3.4):

    S = γ·S_TTD + (1−γ)·S_CM,  γ = 0.5
    S_TTD = 1 − mean(TTD_i / Δt_i)      time-to-detection
    S_CM  = (TPR + TNR) / 2             balanced accuracy, not F1

`batadal_score.py` implements this. Measured:

| config | S_TTD | TPR | TNR | S_CM | **S** | F1 | detected |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conservative | 0.216 | 0.209 | 0.875 | 0.542 | **0.379** | 0.251 | 2/7 |
| Balanced | 0.314 | 0.152 | 0.947 | 0.550 | **0.432** | 0.225 | 4/7 |
| Aggressive | 0.202 | 0.012 | 0.987 | 0.500 | **0.351** | 0.023 | 3/7 |

**S = 0.432 at best. That is below every published BATADAL entry** — the
lowest ranked, Aghashahi, scored 0.534; the winner, Housh & Ohar, 0.970. An
`S_CM` of 0.550 is barely above the 0.500 that a naive always-safe predictor
earns. This detector is weaker than the field by the field's own measure, and
the repo should be read as a reproducible baseline, not a working detector.

### REFUTED: "F1 is the wrong metric, the official one will be kinder"

Before running the scorer it was argued that F1 unfairly penalises a
*change* detector — a two-window method can only fire near a transition, so
tick-level recall counts as misses the ticks it is mathematically unable to
alarm on. The prediction was that the competition metric would rank the
detector substantially better.

**It does not.** S = 0.432 still sits below last place. The reasoning about
change detectors is sound and is demonstrated in `math_audit.py`, but it does
not rescue the result. Prediction registered, prediction refuted, kept.

### Two things the metric change did reveal

**The best configuration flips.** F1 ranks Conservative highest (0.251); S
ranks Balanced highest (0.432) and demotes Conservative to 0.379. Shipping on
the F1 headline would have shipped the wrong configuration.

**Time-to-detection is the strongest dimension.** When it fires it is fast:

| attack | duration | TTD | detected |
|---:|---:|---:|---|
| 1 | 70 h | 59 h | yes |
| 2 | 65 h | 21 h | yes |
| 3 | 31 h | 12 h | yes |
| 4 | 31 h | — | **no** |
| 5 | 100 h | 25 h | yes |
| 6 | 80 h | — | **no** |
| 7 | 30 h | — | **no** |

---

## 2026-07-27 — Self-test suite added (Claude, Anthropic; sandbox)

`test_smoke.py` — 11 dataset-free self-tests, NumPy only, seconds to run.

```
S1  jsd(h,h) == 0 for identical histograms                    0.000e+00
S2  jsd symmetric, bounded in [0,1] (log2)                    max 0.2922
S3  disjoint histograms score above overlapping               1.000 > 0.156
S4  episodes() segmentation, incl. edges and singletons       5 cases
S5  evaluate() F1 1.0 perfect / 0.0 silent                    1.000 / 0.000
S6  warm-up mask excludes early ticks                         recall 0 inside warm-up
S7  ANTI-VACUITY: flat input -> 0, shift -> 7.19x floor       floor 0.0887 signal 0.6378
S8  calibrate() percentile; empty input -> 0.0                thr 0.9950 / 0.0
S9  loader counts unparseable cells                           clean 0/0, dirty 2/1
S10 attribution ranks the injected channel top-5              target 1, area 1
S11 KNOWN LIMIT: window=10 noise floor swamps signal          ratio 0.95x

11/11 self-tests passed
```

`python3 test_smoke.py --sabotage` replaces `jsd` with a constant and reports
0/2, exit 1. A suite never observed failing is not evidence of anything.

These tests do **not** verify the five BATADAL numbers — that needs the real
dataset, logged above.

---

## 2026-07-27 — Mathematical audit (`math_audit.py`)

16 properties. **All six JSD theorems hold**, including that √JSD satisfies
the triangle inequality (Endres & Schindelin 2003) — zero violations in 20,000
random triples. Non-negativity, exact symmetry, boundedness with disjoint
support attaining 1.0, identity of indiscernibles, and rescaling invariance to
1.9e-11. The divergence implementation is correct.

Three hypothesised flaws were **refuted** and are kept: the score does not
collapse for large attacks (an emptied histogram normalises to uniform, which
still sits 0.829 from a populated one), no evasion was found at any magnitude
up to 10,000, and channels constant during training do not go blind.

Three real limits were found:

- **A sustained attack is only detectable near its onset.** Once it is older
  than 2W ticks both windows sit inside the attacked regime and the score
  returns to the noise floor. Coverage of a 400-tick synthetic attack: 18.5%
  at W=30, 43.5% at W=60, 40.2% at W=100.
- **Tick recall and episode detection disagree structurally** — on identical
  synthetic runs, 48.8% versus 30/30.
- **Signal-to-noise depends strongly on window size**, and the ordering
  explains the published recall ordering:

| window | bins | noise floor | signal | ratio |
|---:|---:|---:|---:|---:|
| 10 | 8 | 0.6786 | 0.6432 | **0.95** |
| 30 | 16 | 0.3631 | 0.6987 | 1.92 |
| 60 | 16 | 0.1926 | 0.7607 | 3.95 |
| 100 | 12 | 0.0785 | 0.6357 | **8.09** |

At W=10 the detector's own sampling noise exceeds a 50-sigma attack.

These are synthetic — Gaussian noise plus level shifts. They establish
properties of the algorithm, not performance on BATADAL's physical attacks.

---

## Data handling: `ATT_FLAG = -999`

The BATADAL rules define `-999` as **unlabeled**. `float("-999")` parses
cleanly and is less than 0.5, so the original loader silently scored every
such row as "not under attack". `dataset04.csv` contains **3,958 of them**, and
the organisers' own table shows attacks 2 and 7 — 24 h and 110 h — with zero
labelled hours. Anyone developing against dataset04 was training on thousands
of fabricated negatives, and any TNR or FPR computed from it was optimistic.

The loader now counts and reports these separately. Note the coercion counter
alone would never have caught it: nothing raised an exception.

`attacks.csv` transcribes the organisers' attack tables. Its `labeled_hours`
column sums to 219, which matches the measured count of `ATT_FLAG == 1` rows
in dataset04 exactly — an independent check on the transcription.

---

## Open items

- [ ] Confirm `fetch_batadal.sh` on a clean clone. The test file published on
      batadal.net is the original unlabelled release with no `ATT_FLAG`
      column; only the mirror carries labels. The script now verifies the
      column is present and exits non-zero if it is not.
- [ ] Independent reproduction by someone who is not the author.

## Reproduced it?

Open an issue with your Python version, NumPy version, platform, the five
numbers, the `data integrity` line, and your `batadal_score.py` output.
Mismatches are more useful than matches and will be logged here.

## Data source and citation

Data is not redistributed here. Run `fetch_batadal.sh`, or see
<https://batadal.net/data.html>.

> R. Taormina, S. Galelli, N. O. Tippenhauer, A. Ostfeld, E. Salomons,
> "BATtle of the Attack Detection ALgorithms (BATADAL): Detailed Problem
> Description and Rules", 9 September 2016. <http://www.batadal.net>
