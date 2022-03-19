#!/bin/bash

for ((i=0;i<10;i+=1))
do 
	python3 src/main.py \
    --env_name Reacher-v2 \
	--seed $i

	python3 src/main.py \
	--mer \
    --env_name Reacher-v2 \
	--seed $i

		python3 src/main.py \
    --env_name Hopper-v3 \
	--seed $i

	python3 src/main.py \
	--mer \
    --env_name Hopper-v3 \
	--seed $i

		python3 src/main.py \
    --env_name HalfCheetah-v3 \
	--seed $i

	python3 src/main.py \
	--mer \
    --env_name HalfCheetah-v3 \
	--seed $i
done

