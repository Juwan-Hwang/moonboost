#!/bin/bash
# A/B split_penalty at a fixed config across datasets
cd ~/mb/acc && source ../venv/bin/activate
export MB_BIN=~/mb/mbx/_build/native/release/build/cmd/main/main.exe
CFG="--grow-policy lossguide --depth 0 --max-leaves 255 --max-cat-to-onehot 32 --min-data-per-group 10"
for d in adult amazon bank click credit electricity phoneme higgs; do
  for p in 0 0.5 1 2; do CORE=${CORE:-0} python ds_eval.py $d $CFG --split-penalty $p; done
done
