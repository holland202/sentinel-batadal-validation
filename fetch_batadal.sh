#!/usr/bin/env bash
# Fetch the 3 public BATADAL CSVs (not redistributed in this repo).
set -e
mkdir -p batadal && cd batadal
MIR="https://raw.githubusercontent.com/SYChen123/Baseline-outlier-detection-algorithms-on-BATADAL-dataset/master/data"
for f in dataset03 dataset04 test_dataset; do
  curl -sfL "https://www.batadal.net/data/BATADAL_$f.csv" -o "$f.csv" \
  || curl -sfL "$MIR/$f.csv" -o "$f.csv" || { echo "FAILED $f"; exit 1; }
  echo "got $f.csv"
done
wc -l *.csv
