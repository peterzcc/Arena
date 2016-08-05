#!/bin/bash

for lr in 0.001 0.01 0.1
do
  python main.py --init_lr $lr
done