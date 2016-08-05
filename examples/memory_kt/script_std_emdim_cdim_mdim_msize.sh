#!/bin/bash

for memory_size in 128 256 1024
do
    for embed_dim in 100 200 50
    do
        for control_state_dim in 100 10 50
        do
            for memory_state_dim in 100 10 50
            do
                python main.py --gpus 0 --memory_size $memory_size --embed_dim $embed_dim --control_state_dim $control_state_dim --memory_state_dim $memory_state_dim
            done
        done
    done
done