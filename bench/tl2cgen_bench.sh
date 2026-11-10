#!/bin/bash
# TL2cgen (Treelite's C code generator) on the XGBoost models in bench/data,
# with and without threshold quantization, compiled with gcc -O3
# -march=native in 64 translation units. Needs treelite and tl2cgen
# installed for python3, and gcc and make.
# Run from bench/:  ./tl2cgen_bench.sh [small medium large]
set -e
mkdir -p data/tl2cgen
for m in ${@:-small medium large}; do
  for q in 0 1; do
    d=data/tl2cgen/${m}_q$q
    python3 - "$m" "$q" "$d" <<'PY'
import sys, treelite, tl2cgen
m, q, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
model = treelite.frontend.load_xgboost_model(f"data/{m}.json")
tl2cgen.generate_c_code(model, dirpath=d, params={"parallel_comp": 64, "quantize": q})
PY
    s=$(date +%s)
    printf 'all: %s\n%%.o: %%.c\n\tgcc -O3 -march=native -I. -c $< -o $@\n' \
      "$(cd $d && ls *.c | sed 's/\.c$/.o/' | tr '\n' ' ')" > $d/Makefile
    (cd $d && make -s -j"$(nproc)" && gcc -O3 -march=native -I. -o bench.exe *.o ../../../tl2cgen_bench.c -lm)
    echo "$m quantize=$q (compiled in $(( $(date +%s) - s )) s): $($d/bench.exe data/input.csv data/$m.expected.csv 5)"
  done
done
