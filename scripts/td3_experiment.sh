#!/bin/bash

for ((i = 0; i < 10; i += 1)); do
  for environment in HalfCheetah-v4 Hopper-v4 Walker2d-v4 Ant-v4 Reacher-v4 Pusher-v4 InvertedDoublePendulum-v4; do
    python3 src/main.py \
    --env_name $environment \
    --seed $i \
    --number_of_rbs 1 \
    --wandb
  done
done
