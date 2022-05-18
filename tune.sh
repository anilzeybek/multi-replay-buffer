#!/bin/bash

# 5, 10000, 0.8

for s in 0 1; do
  python src/main.py --seed=$s --number_of_rbs=1

  for norb in 5 9; do
    for cf in 10000 25000; do
      for alpha in 0.6 0.8 0.9; do
        python src/main.py --seed=$s --number_of_rbs=$norb --clustering_freq=$cf --alpha=$alpha
      done
    done
  done
done
