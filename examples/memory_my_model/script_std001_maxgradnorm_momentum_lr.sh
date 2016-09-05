#!/bin/bash

for std in 0.1 0.01 0.05
do
    for maxgradnorm in 100 50 10
    do
        for momentum in 0.5 0.8 0.9
        do
            for lr in 0.1 0.01 0.001
            do
                python main.py --gpus 0 --init_lr $lr --momentum $momentum --init_std $std --maxgradnorm $maxgradnorm
            done
        done
    done
done