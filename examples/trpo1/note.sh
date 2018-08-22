#!/usr/bin/env bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- bash

CUDA_VISIBLE_DEVICES=3 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" python3 trpo_test.py --nactor 20 --batch-size 20000

tmux kill-window -t 9

sshfs dy2:/home/data/czeng/ ~/dy2

LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip3 install --user --upgrade .

CUDA_VISIBLE_DEVICES=3 python3 trpo_test.py --nactor 20

pip3 freeze | grep -v "^-e" | xargs pip3 uninstall -y

from pympler import muppy, summary
all_objects = muppy.get_objects()
len(all_objects)
sum1 = summary.summarize(all_objects)
summary.print_(sum1)

summary.print_(summary.summarize(muppy.get_objects()))

pip3.6 --no-cache-dir install --user --force-reinstall https://github.com/mind/wheels/releases/download/tf1.4-gpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
tmu

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON

import cv2;cv2.imshow('ob',obs[1][0]);cv2.waitKey(0);cv2.destroyAllWindows()


img=img_input[0,:,:,0]
import cv2;cv2.imshow('ob',img);cv2.waitKey(0);cv2.destroyAllWindows()

from matplotlib import pyplot as plt;plt.figure(1);plt.imshow(obs[1][0,:,:,0], cmap='gray', interpolation='bicubic');plt.xticks([]), plt.yticks([]);plt.draw()


python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON \
-DPYTHON_INCLUDE_DIRS=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARIES=$(python3.6 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))")



sudo ifconfig wlp2s0 down && sudo ifconfig wlp2s0 up

while true; do
if ! ping -q -w 1 -c 1 `ip r | grep default | cut -d ' ' -f 3` > /dev/null; then
nmcli --wait 20 c up eduroam
echo connecting
else
echo connected
fi;
sleep 2;
done
nvidia-smi --query-gpu="utilization.gpu,memory.free,memory.used" --format=csv -lms 500
nvidia-smi --query-gpu="accounting.mode" --format=csv

nvidia-smi -q -d "MEMORY,UTILIZATION" -lms 500
nvidia-smi --query-accounted-apps="gpu_utilization" --format=csv

CUDA_VISIBLE_DEVICES=0 disprun python3.6 pg_train.py --env dynamic2d --rl-method PG --nactor 20 --batch-size 20480 --withimg 1 --nfeat 30 --load-dir models --load-model 0 --vlr 0.002 --load-leaf 1 --train-leaf 0 --train-decider 1 --train-switcher 0 --switcher-start 150  --switcher-length 25 --npret -1  --loss TRAD --ent-k 0.01 --switcher-k 1.0  --lr 0.0003
disprun python3.6 ~/Arena/examples/trpo1/plot.py --dir ../exp_23 ../exp_24 ../exp_25 ../exp_26 --label 0.0001 0.0003 0.001 0.003
python3.6 $M2WS/Arena/examples/trpo1/visualize_log.py --dataname std
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname mean_r_t --extra move0 move1 move2 move3 move4 move5 move6 move7 -w200
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname std
disprun python3.6 ~/Arena/examples/trpo1/plot.py -w20
tmux -L s2 new -s 2
CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00001 --vlr 0.001 --npass 2 --loss TRAD_WASS --decrease-with-initial 0.03 --wass-decrease-period 30e6
CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001 --npass 2 --loss TRAD_WASS --decrease-with-initial 0.03 --wass-decrease-period 30e6
CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001 --npass 2 --loss TRAD_WASS --decrease-with-initial 0.03 --wass-decrease-period 30e6
disprun python3.6 ~/Arena/examples/trpo1/plot.py -w50 --dir ../exp_20 ../exp_21 ../exp_22 --label 1e-5 3e-5 1e-4
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname std --dir ../exp_20 ../exp_21 ../exp_22 --label 1e-5 3e-5 1e-4
WS=
python3.6 ${WS}/plot.py -w20 --dir ../rec_0403_move1_fail ../rec_0403_move2 --label run1 run2

python3.6 pg_train.py --env CartPole-v1 --norm-gae 0 --rl-method PG --nactor 16 --batch-size 9600 --withimg 0 --nfeat 0 --load-model 0 --vlr 0.001 --npass 1 --minibatch-size 128 --lr 0.0001 --load-leaf 0 --train-leaf 0 --train-decider 1 --train-switcher 0 --switcher-length 1 --npret -1  --loss TRAD

disprun python3.6 ~/Arena/examples/trpo1/plot.py -w200 --dir ../exp_35
tmux send-keys -t 0 "echo 'OK'"

CUDA_VISIBLE_DEVICES=0,2,3 ./contact_czeng_if_you_need_gpu.sh
CUDA_VISIBLE_DEVICES=0,3 ./attack_gpu.sh

Reacher-v1
HalfCheetah-v1
Hopper-v1
Swimmer-v1
bad Walker2d-v1
ongoing Humanoid-v1
InvertedPendulum-v1
bad InvertedDoublePendulum-v1
disprun python3.6 $WS/plot.py -w200

/home/peterzeng/dy2/Arena/examples/trpo1/rec_180809_constdirreachreg_decider_1
0mux1
5
use dead CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env Reacher-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
use dead CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env HalfCheetah-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
use dead CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env Hopper-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
use dead CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env Swimmer-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
daed CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env Walker2d-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
10
use dead CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env Humanoid-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
use dead CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env InvertedPendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
dead CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
12
disprun python3.6 $WS/plot.py -w20

1mux1
dead CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env Walker2d-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env Humanoid-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
use dead CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env Walker2d-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env Humanoid-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env Walker2d-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env Humanoid-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
dead CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01

disprun python3.6 $WS/plot.py -w20 --dir ../exp_1 ../exp_4 ../exp_15 --label 1 2 3
disprun python3.6 $WS/plot.py -w20 --dir ../exp_2 ../exp_13 ../exp_16 --label 1 2 3
disprun python3.6 $WS/plot.py -w200 --dir ../exp_3 ../exp_14 ../exp_17 --label 1 2 3

2mux1
CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00001 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
use CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env InvertedDoublePendulum-v1 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.000003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.01
CUDA_VISIBLE_DEVICES=2,3,0,1 disprun python3.6 pg_train.py --env constdirreachreg --rl-method PG --nactor 32 --batch-size 5120 --withimg 1 --nfeat 16 --load-dir models_jointly_trained --load-model 1 --vlr 0.001 --npass 2 --minibatch-size 128 --lr 0.0001 --multi-update 1 --norm-gae 1 --load-leaf 1 --train-leaf 0 --train-decider 1 --train-switcher 1 --switcher-length 10 --npret -1  --loss PPO --switcher-k 0.01 --switcher-time-weight 0.0  --regulation-k 50.0 --min-prob 0.0001

disprun python3.6 $WS/plot.py -w20 --dir ../exp_18 ../exp_19 --label 1 2
disprun python3.6 $WS/plot.py -w200 --dir ../exp_19 -t15

disprun python3.6 $WS/plot.py -w20 --dir ../exp_20
disprun python3.6 $WS/visualize_log.py --dataname subpolicy_len --dir ../exp_20
local:
CUDA_VISIBLE_DEVICES=1 disprun python3.6 pg_train.py --env flatcont2d --rl-method PG --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --lr 0.0001 --multi-update 1 --vlr 0.001 --npass 2 --loss PPO --use-mix true

record:
emacs ~/gym/gym/envs/mujoco/mujoco_env.py
self._get_viewer(visible=False,init_height=640,init_width=640).render()
Reacher-v1
HalfCheetah-v1
Hopper-v1
Swimmer-v1
bad Walker2d-v1
ongoing Humanoid-v1
InvertedPendulum-v1
bad InvertedDoublePendulum-v1
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Reacher-v1 --withimg 1 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0

disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env stateconstdirreachreg --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-length 10
disprun python3.6 pg_train.py --nactor 1 --num-steps 5000 --batch-size 5000 --env flatcont2d --withimg 1 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0

disprun python3.6 $WS/plot.py -w20 --dir ../move1 ../move2 ../move3  --label run1 run2 run3

disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Reacher-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env HalfCheetah-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Hopper-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Swimmer-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Walker2d-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env Humanoid-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env InvertedPendulum-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0
disprun python3.6 pg_train.py --nactor 1 --num-steps 1000 --batch-size 1000 --env InvertedDoublePendulum-v1 --withimg 0 --nfeat 16 --load-model 1 --load-leaf 1 --train-leaf 0 --train-decider 0 --train-switcher 0 --render record --load-dir models_test --switcher-time-weight 0.0

LD_PRELOAD="${M2WS}/lib/faketime/libfaketime.so.1" FAKETIME="2020-01-01" bash

rsync -avP dy2:/home/data/czeng/Arena/examples/trpo1/rec_180810_constdirreachreg_switcher_good .
/home/peterzeng/dy2/Arena/examples/trpo1//home/peterzeng/dy2/Arena/examples/trpo1/rec_180810_constdirreachreg_switcher_good
disprun python3.6 $WS/plot.py -w20 --dir ../exp_27 ../exp_28 --label w=3e-3 w=1e-2 -a0.1
python3.6 $WS/plot.py -w200
python3.6 $WS/visualize_log.py --dataname subpolicy_len

detex thesis.tex > thesis.txt
i +=0

i +=1;plt.figure(0);plt.imshow(x[i,:,:],cmap="gray");plt.figure(1);plt.imshow(recon[i,:,:,0],cmap="gray");plt.figure(2);plt.imshow(np.reshape(features[i,:,:,:],(7,7,1)),cmap="gray")

python3.6 ../plot.py -w 20 --dir "../rec_dy2/rec_180505_move0" "."

CUDA_VISIBLE_DEVICES="" python3.6 ./project2.py --task train_ae

i += 1;visualize_ae(i,x,features,recon)

hist = np.histogram(sampled_y_test,bins=47)[0]


