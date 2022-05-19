#!/bin/bash

for environment in HalfCheetah-v3 Hopper-v3 Ant-v2 Reacher-v2 Pusher-v2 InvertedDoublePendulum-v2; do
  echo '--' $environment >>result.txt

  echo -n "orig: " >>result.txt
  python3 src/main.py \
    --env_name $environment \
    --number_of_rbs 1 \
    --test

  echo -n "mer: " >>result.txt
  python3 src/main.py \
    --env_name $environment \
    --number_of_rbs 5 \
    --test
done
