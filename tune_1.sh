#!/bin/bash

for environment in HalfCheetah-v3 Hopper-v3 Walker2d-v3 Reacher-v2 InvertedDoublePendulum-v2; do
    for norb in 17 33 65 127; do
        for alpha in 0.6 0.8; do
            for s in 0 1 2; do
                python src/main.py --env_name=$environment --seed=$s --number_of_rbs=$norb --normalize_is --alpha=$alpha
            done

            echo $environment '--' $norb '-- niTrue --' $alpha >>result_1.txt
            python src/main.py --env_name=$environment --test --file_name=result_1.txt --number_of_rbs=$norb --normalize_is --alpha=$alpha
        done
    done
done
