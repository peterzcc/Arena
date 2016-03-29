#!/bin/bash
#$ -S /bin/bash
#source ~/.bashrc
. /project/dygroup2/czeng/venv/bin/activate
name=breakout
n=2
s=2
# workerq="*.q@client111,*.q@client112,*.q@client113,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110"
# serverq="*.q@client111,*.q@client112,*.q@client113,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110"
workerq="*.q@client112,*.q@client114,*.q@client113,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110"
serverq="*.q@client112,*.q@client114,*.q@client113,*.q@client114,*.q@client115,*.q@client108,*.q@client109,*.q@client110"

wd="/csproject/dygroup2/czeng/dist_dqn/"
activate_cmd="./project/dygroup2/czeng/venv/bin/activate"
script="python dqn_dist_demo.py  --double-q 1 -c gpu1 -r roms/breakout.bin -eps 0.1 --kv-type dist_async"


python traker/dmlc_sge.py --activate-cmd ${activate_cmd} --log-file ${name}_n${n}_s${s}.out -wd ${wd} --jobname dmlc-${name}-n${n}-s${s} -workerq ${workerq} -serverq ${serverq} -n ${n} -s ${s} $script
