#!/bin/bash

for ((i=0;i<10;i+=1))
do 
	python3 src/main.py \
    --env_name LunarLanderContinuous-v2 \
	--seed $i

	python3 src/main.py \
	--mer \
    --env_name LunarLanderContinuous-v2 \
	--seed $i
done