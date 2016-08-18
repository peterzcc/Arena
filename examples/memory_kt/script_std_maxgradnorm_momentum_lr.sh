#!/bin/bash
for k in 1 2 5
do
    for dim in 50 20 10
    do
        for std in 0.01 0.05 0.1
        do
            for maxgradnorm in 10.0 50.0 100.0
            do
                for momentum in 0.9 0.8
                do
                    for lr in 0.1
                    do
                        python main.py --gpus 0 --init_lr $lr --momentum $momentum --init_std $std --maxgradnorm $maxgradnorm --embed_dim $dim --control_state_dim $dim memory_state_dim $dim --k_smallest $k
                    done
                done
            done
        done
    done
done