#!/bin/bash

for std in 0.05 0.01 0.1
do
    for maxgradnorm in 10 50 100
    do
        for momentum in 0.9 0.8 0.5
        do
            for lr in 0.1
            do
                python main.py --gpus 0 --init_lr $lr --momentum $momentum --init_std $std --maxgradnorm $maxgradnorm
            done
        done
    done
done