#!/bin/bash
# abt.sh base.exe new.exe N core -- extra args : alternate timing, airline Xh
B=$1; C=$2; N=$3; K=$4; shift 5
cd ~/mb/acc; A="Xh.csv y.csv --objective binary:logistic --rounds 100 --eta 0.1 $*"
for i in $(seq $N); do
  echo "base $(taskset -c $K $B train $A -o /tmp/ab_b$K.json | grep -o '[0-9]* ms')  new $(taskset -c $K $C train $A -o /tmp/ab_n$K.json | grep -o '[0-9]* ms')"
done
cmp -s /tmp/ab_b$K.json /tmp/ab_n$K.json && echo SAME || echo DIFF
