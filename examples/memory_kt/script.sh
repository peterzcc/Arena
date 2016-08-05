#!/bin/bash

for std in 0.05 0.01 0.1
    for maxgradnorm in 10 50 100
        for lr in 0.001 0.01 0.1
        do
            python main.py --gpus 0 --init_lr $lr --init_std $std --maxgradnorm $maxgradnorm
        done
    done
done