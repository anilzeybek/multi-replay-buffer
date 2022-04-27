#!/bin/bash

for ((i=0;i<5;i+=1))
do 
	python3 src/main.py \
	--test \
    --env_name Reacher-v2 \
	--seed $i

	python3 src/main.py \
	--test \
	--mer \
    --env_name Reacher-v2 \
	--seed $i

	python3 src/main.py \
	--test \
    --env_name Hopper-v3 \
	--seed $i

	python3 src/main.py \
	--test \
	--mer \
    --env_name Hopper-v3 \
	--seed $i

	python3 src/main.py \
	--test \
    --env_name HalfCheetah-v3 \
	--seed $i

	python3 src/main.py \
	--test \
	--mer \
    --env_name HalfCheetah-v3 \
	--seed $i

	python3 src/main.py \
	--test \
    --env_name Ant-v2 \
	--seed $i

	python3 src/main.py \
	--test \
	--mer \
    --env_name Ant-v2 \
	--seed $i

	python3 src/main.py \
	--test \
    --env_name Pusher-v2 \
	--seed $i

	python3 src/main.py \
	--test \
	--mer \
    --env_name Pusher-v2 \
	--seed $i
done
