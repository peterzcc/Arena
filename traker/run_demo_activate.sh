#!/bin/bash
#$ -S /bin/bash
#source ~/.bashrc

name=cifar10
n=5
s=5
workerq="all.q@client111,all.q@client112,all.q@client113,all.q@client114,all.q@client115"
serverq="all.q@client111,all.q@client112,all.q@client113,all.q@client114,all.q@client115"
wd="/csproject/dygroup2/czeng/venv/downloads/mxnet/example/image-classification/"
activate_cmd=". /csproject/dygroup2/czeng/venv/bin/activate"
script="python /csproject/dygroup2/czeng/venv/downloads/mxnet/example/image-classification/train_cifar10.py --kv-store dist_sync"


#Download CIFAR10 data if no data directory found

if [ ! -d "cifar10" ]; then
    mkdir cifar10
    cd cifar10
    wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip
    unzip -u cifar10.zip
    mv cifar/* . && rm -rf cifar && rm cifar10.zip
    cd ..
fi

python dmlc_sge.py --activate-cmd ${activate_cmd} --log-file ${name}_n${n}_s${s}.out -wd ${wd} --jobname dmlc-${name}-n${n}-s${s} -workerq ${workerq} -serverq ${serverq} -n ${n} -s ${s} $script
