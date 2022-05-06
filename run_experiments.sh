#!/bin/bash

for ((i=0;i<5;i+=1))
do 
	for environment in HalfCheetah-v3 Hopper-v3 LunarLanderContinuous-v2 Ant-v2 Reacher-v2 Pusher-v2 InvertedDoublePendulum-v2;
	do
		for norb in 1 5 
		do
			python3 src/main.py \
			--env_name $environment \
			--seed $i \
			--number_of_rbs $norb

			python3 src/main.py \
			--env_name $environment \
			--seed $i \
			--number_of_rbs $norb \
			--test
		done
	done
done
