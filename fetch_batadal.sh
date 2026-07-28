#!/usr/bin/env bash
# Fetch the public BATADAL CSVs. The data is NOT redistributed in this repo.
#
# Source: BATtle of the Attack Detection ALgorithms (BATADAL)
#   R. Taormina, S. Galelli, N.O. Tippenhauer, A. Ostfeld, E. Salomons et al.
#   http://www.batadal.net  ·  https://batadal.net/data.html
#
# WHY THIS SCRIPT VERIFIES ITS OWN OUTPUT
# The test dataset published on batadal.net is the ORIGINAL COMPETITION
# RELEASE and has NO ATT_FLAG column -- the labels were released separately
# after the submission deadline. A mirror carries the labelled version. If we
# fetched blindly, a cold clone could land the unlabelled file and
# sentinel_batadal.py would die with `ValueError: 'ATT_FLAG' is not in list`.
# So every file is checked for the columns it must have, and the script exits
# non-zero if a check fails. A fetch script that cannot fail is not a fetch
# script, it is a wish.
set -euo pipefail

MIR="https://raw.githubusercontent.com/SYChen123/Baseline-outlier-detection-algorithms-on-BATADAL-dataset/master/data"
mkdir -p batadal && cd batadal

fetch_one() {
  local f="$1" need_flag="$2"
  rm -f "$f.csv"
  curl -sfL "https://www.batadal.net/data/BATADAL_$f.csv" -o "$f.csv" 2>/dev/null || true

  if [ ! -s "$f.csv" ] || { [ "$need_flag" = "yes" ] && ! head -1 "$f.csv" | grep -q "ATT_FLAG"; }; then
    echo "  primary source unusable for $f, trying mirror"
    curl -sfL "$MIR/$f.csv" -o "$f.csv" || { echo "FAILED to fetch $f"; exit 1; }
  fi

  [ -s "$f.csv" ] || { echo "FAILED: $f.csv is empty"; exit 1; }

  if [ "$need_flag" = "yes" ] && ! head -1 "$f.csv" | grep -q "ATT_FLAG"; then
    echo "FAILED: $f.csv has no ATT_FLAG column from either source."
    echo "        The reference run needs labels. See batadal.net/data.html."
    exit 1
  fi
  echo "  ok $f.csv  ($(wc -l < "$f.csv") lines)"
}

echo "fetching BATADAL data (not redistributed in this repo)"
fetch_one dataset03   yes    # clean training data, no attacks
fetch_one test_dataset yes   # test set, 7 attacks -- REQUIRES labels
fetch_one dataset04   yes    # development set: 7 attacks, only 219 hours
                             # labelled; 3958 rows are ATT_FLAG = -999
                             # (UNLABELED, not safe). Not used by the
                             # reference run. See REPRODUCED.md.

echo
echo "verification:"
for f in dataset03 test_dataset dataset04; do
  printf "  %-14s %6s rows, ATT_FLAG present\n" "$f" "$(($(wc -l < "$f.csv") - 1))"
done
echo
echo "done. now run:  python3 sentinel_batadal.py ./batadal"
