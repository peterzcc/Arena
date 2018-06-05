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


CUDA_VISIBLE_DEVICES=1 python3 trpo_test.py --nactor 20 --batch-size 10000

python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON \
-DPYTHON_INCLUDE_DIRS=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARIES=$(python3.6 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))")

python3 -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))"

python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"

java -jar ../pp2check/jplag.jar -l python3 -s . -r 0result -bc 0base -m 30%

registeruser
mail peterzengcc@gmail.com

for d in */ ; do
    mv $d/*/**/*(.D) $d
done

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

CUDA_VISIBLE_DEVICES=0,1,2 ./contact_czeng_if_you_need_gpu.sh
0mux1
dead CUDA_VISIBLE_DEVICES=1 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR_ADAM --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.00003 --vlr 0.001 --npass 2 --loss TRAD_WASS --use-mix true --decrease-with-initial 0.005 --wass-decrease-period 5e6
dead CUDA_VISIBLE_DEVICES=2 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR_ADAM --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.001 --npass 2 --loss TRAD_WASS --use-mix true --decrease-with-initial 0.005 --wass-decrease-period 5e6
disprun python3.6 ~/Arena/examples/trpo1/plot.py -w200 --dir ../exp_19 ../exp_18  --label 3e-5 3e-4
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname std --dir ../exp_19 ../exp_18  --label 3e-5 3e-4



1mux1
#flat 2d
CUDA_VISIBLE_DEVICES=3 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.002 --loss TRAD --lr 0.001 --stdbias 0.0
#CUDA_VISIBLE_DEVICES=0 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.002 --loss TRAD --lr 0.001 --stdbias 0.02
#CUDA_VISIBLE_DEVICES=1 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.002 --loss TRAD --lr 0.001 --stdbias 0.005
CUDA_VISIBLE_DEVICES=2 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.002 --loss TRAD --lr 0.001 --stdbias 0.01
CUDA_VISIBLE_DEVICES=0 disprun python3.6 pg_train.py --env flatcont2d --rl-method ACKTR --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.0003 --vlr 0.002 --loss TRAD --lr 0.001 --stdbias 0.04
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname std --dir ../exp_87 ../exp_90 ../exp_100 ../exp_102 --label 0 0.01 0.02 0.04
disprun python3.6 ~/Arena/examples/trpo1/plot.py -w32 --dir ../exp_87 ../exp_90 ../exp_100 ../exp_102 --label 0 0.01 0.02 0.04

2mux1
1 decrease
CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001  --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02
CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001 --npass 2 --loss TRAD_WASS --decrease-with-initial 0.02

4
CUDA_VISIBLE_DEVICES=0 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0003 --vlr 0.001 --npass 2 --loss TRAD_WASS
disprun python3.6 ~/Arena/examples/trpo1/plot.py -w200 --dir ../exp_10 ../exp_11 ../exp_7 ../exp_8 ../exp_9 --label de3e-5 de1e-4 1e-4 3e-4 1e-3
disprun python3.6 ~/Arena/examples/trpo1/visualize_log.py --dataname std --dir ../exp_10 ../exp_11 ../exp_7 ../exp_8 ../exp_9 --label de3e-5 de1e-4 1e-4 3e-4 1e-3

5
CUDA_VISIBLE_DEVICES=2 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.00003 --vlr 0.001 --npass 2 --loss TRAD_WASS --normalize-wass 1
CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0001 --vlr 0.001 --npass 2 --loss TRAD_WASS --normalize-wass 1
CUDA_VISIBLE_DEVICES=3 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.0003 --vlr 0.001 --npass 2 --loss TRAD_WASS --normalize-wass 1
CUDA_VISIBLE_DEVICES=1 python3.6 pg_train.py --env move0 --rl-method ACKTR_ADAM --nactor 20 --batch-size 4000 --withimg 0 --nfeat 0 --load-model 0 --kl 0.001 --vlr 0.001 --npass 2 --loss TRAD_WASS --normalize-wass 1



local:
python3.6 pg_train.py --env flatcont2d --rl-method ACKTR_ADAM --nactor 32 --batch-size 2560 --withimg 1 --nfeat 16 --load-model 0 --kl 0.001 --vlr 0.001 --npass 2 --loss TRAD_WASS --use-mix true --decrease-with-initial 0.01 --wass-decrease-period 30e6

record:
CUDA_VISIBLE_DEVICES="" python3.6 pg_train.py --nactor 1 --num-steps 5000 --batch-size 400 --withimg 1 --env simplemove1d --nfeat 30 --load-model 1 --no-train 1 --save-model 0 --render record --load-dir exp_7

LD_PRELOAD="${M2WS}/lib/faketime/libfaketime.so.1" FAKETIME="2020-01-01" bash

import matplotlib.pyplot as plt

i +=0

i +=1;plt.figure(0);plt.imshow(x[i,:,:],cmap="gray");plt.figure(1);plt.imshow(recon[i,:,:,0],cmap="gray");plt.figure(2);plt.imshow(np.reshape(features[i,:,:,:],(7,7,1)),cmap="gray")

python3.6 ../plot.py -w 20 --dir "../rec_dy2/rec_180505_move0" "."

CUDA_VISIBLE_DEVICES="" python3.6 ./project2.py --task train_ae

i += 1;visualize_ae(i,x,features,recon)

hist = np.histogram(sampled_y_test,bins=47)[0]


