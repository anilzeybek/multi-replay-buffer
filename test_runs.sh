#!/bin/bash

echo "orig" >> result.txt

for environment in HalfCheetah-v3 Hopper-v3 LunarLanderContinuous-v2 Ant-v2 Reacher-v2 Pusher-v2 InvertedDoublePendulum-v2;
do 
    echo '--' $environment >> result.txt

    for ((i=0;i<5;i+=1))
	do
        python3 src/main.py \
        --env_name $environment \
        --seed $i \
        --number_of_rbs 1 \
        --test
	done
done

echo "" >> result.txt
echo "mer" >> result.txt

for environment in HalfCheetah-v3 Hopper-v3 LunarLanderContinuous-v2 Ant-v2 Reacher-v2 Pusher-v2 InvertedDoublePendulum-v2;
do 
    echo '--' $environment >> result.txt

    for ((i=0;i<5;i+=1))
	do
        python3 src/main.py \
        --env_name $environment \
        --seed $i \
        --number_of_rbs 5 \
        --test
	done
done
