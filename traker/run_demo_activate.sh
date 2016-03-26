#!/bin/bash
#$ -S /bin/bash
#source ~/.bashrc
. /project/dygroup2/czeng/venv/bin/activate
name=breakout
n=2
s=2
# workerq="all.q@client111,all.q@client112,all.q@client113,all.q@client114,all.q@client115,all.q@client108,all.q@client109,all.q@client110"
# serverq="all.q@client111,all.q@client112,all.q@client113,all.q@client114,all.q@client115,all.q@client108,all.q@client109,all.q@client110"
workerq="all.q@client112,all.q@client112,all.q@client113,all.q@client114,all.q@client115,all.q@client108,all.q@client109,all.q@client110"
serverq="all.q@client112,all.q@client112,all.q@client113,all.q@client114,all.q@client115,all.q@client108,all.q@client109,all.q@client110"

wd="/csproject/dygroup2/czeng/dist_dqn"
activate_cmd="./project/dygroup2/czeng/venv/bin/activate"
script="python /csproject/dygroup2/czeng/dist_dqn/dqn_dist_demo.py  --double-q 1 -c gpu0 -r roms/breakout.bin -eps 0.1"


python dmlc_sge.py --activate-cmd ${activate_cmd} --log-file ${name}_n${n}_s${s}.out -wd ${wd} --jobname dmlc-${name}-n${n}-s${s} -workerq ${workerq} -serverq ${serverq} -n ${n} -s ${s} $script
