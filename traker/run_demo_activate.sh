#!/bin/bash
#$ -S /bin/bash
#source ~/.bashrc
. /project/dygroup2/czeng/venv/bin/activate
name=breakout
# nworker=4
# nserver=4

# workerq="*.q@client111,*.q@client112,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110,*.q@client111,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110"
# serverq="*.q@client111,*.q@client112,*.q@client114,*.q@client115"
# CTX="gpu0,gpu1,gpu1,gpu0,gpu1,gpu1,gpu1,gpu0,gpu0,gpu0,gpu0,gpu0,gpu0,gpu0,gpu0"
wd="/csproject/dygroup2/czeng/dist_dqn/"
activate_cmd="./project/dygroup2/czeng/venv/bin/activate"
# script="python easgd_async.py -r roms/breakout.bin -d easgdfixt --replay-start-size 90 --optimizer rmspropnoncentered --lr 0.001 --eps 0.01 --symbol nips --sample-policy recent --rms-decay 0.99 --nactor 16 --single-batch-size 5 --param-update-period 4 --eps-update-period 1000 --kv-type dist_async --easgd-update-period 0.2 --kvstore-update-period 8 --easgd-beta 0.3 --nworker ${n}"
# script="python dqn_async.py -r roms/breakout.bin --replay-start-size 90 --optimizer rmspropnoncentered --lr 0.001 --eps 0.01 --symbol nips --sample-policy recent --rms-decay 0.99 --nactor 16 --single-batch-size 5 --param-update-period 5 --kvstore-update-period 16 --eps-update-period 1000 --kv-type dist_async"

python traker/dmlc_sge.py --activate-cmd ${activate_cmd} --log-file ${name}_n${n}_s${s}.out -wd ${wd} --jobname dmlc-${name}-n${n}-s${s} -workerq ${workerq} -serverq ${serverq} -c ${CTX} -n ${nworker} -s ${nserver} ${script}
