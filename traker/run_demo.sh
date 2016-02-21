name=cifar10
n=1
s=1
workerq="all.q@client111,all.q@client112"
serverq="all.q@client113,all.q@client114"
wd="/project/dygroup2/xingjian/mxnet/example/image-classification/"
script="python2.7 /project/dygroup2/xingjian/mxnet/example/image-classification/train_cifar10.py --kv-store dist_sync"



python2.7 dmlc_sge.py --log-file ${name}_n${n}_s${s}.out -wd ${wd} --jobname dmlc-${name}-n${n}-s${s} -workerq ${workerq} -serverq ${serverq} -n ${n} -s ${s} $script

