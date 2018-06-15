#!/usr/bin/env zsh
initial_data_dir=data_initial
rm -r ${initial_data_dir}
mkdir ${initial_data_dir}
for  ((i=1; i<=8; i++)); do
    i_task=$((i-1))
    i_gpu=$((i_task%4))
    tmux send-keys -t ${i} "CUDA_VISIBLE_DEVICES=${i_gpu} python3.6 pg_train.py --env move${i_task}_task8 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.001 --vlr 0.001 --npass 2 --loss TRAD --initial-state-dir ${initial_data_dir}"
    tmux send-keys -t ${i} "Enter"
    sleep 0.3
done
