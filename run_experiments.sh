#!/bin/bash

for ((i=0;i<10;i+=1))
do 
	python3 src/main.py \
    --env_name Hopper-v3 \
	--solve_score=2000 \
	--seed $i
done

for ((i=0;i<10;i+=1))
do 
	python3 src/main.py \
	--mer \
    --env_name Hopper-v3 \
	--solve_score=2000 \
	--seed $i
done