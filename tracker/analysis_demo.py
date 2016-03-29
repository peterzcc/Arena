import re
import argparse
import sys

parser = argparse.ArgumentParser(description='Demo script to aggregate training statistics from log file')
parser.add_argument('-fl', '--file-list', required=True, type=str,
                    help = 'path of log files, separated by `,`. '
                           'Example: cifar10_n2_s2.out,cifar10_n3_s3.out,cifar10_n4_s4.out,cifar10_n5_s5.out')
parser.add_argument('-o', '--output', default=None, type=str,
                    help = 'path of output file')
args, unknown = parser.parse_known_args()
args.file_list = args.file_list.split(',')
outf = open(args.output, 'w') if args.output is not None else sys.stdout
properties = ['Validation-accuracy', 'Train-accuracy', 'Time cost']
for path in args.file_list:
    training_statistics = {}
    f = open(path, 'r')
    m = re.search('_n(\d+)', path)
    assert m is not None, 'No `_n` found in the file path %s' %(path)
    n = int(m.group(1))
    m = re.search('_s(\d+)', path)
    assert m is not None, 'No `_s` found in the file path %s' %(path)
    s = int(m.group(1))
    for line in f:
        line = line.strip()
        for property in properties:
            m = re.search("Epoch\[(\d+)\] %s=([0-9.]+)" %(property), line)
            if m is not None:
                epoch_id, value = m.groups()
                epoch_id = int(epoch_id)
                if epoch_id not in training_statistics:
                    training_statistics[epoch_id] = {}
                training_statistics[epoch_id][property] = value
            continue
    for epoch_id, statistics in training_statistics.items():
        outf.write("%d %d %d %s\n" %(epoch_id, n, s, ' '.join(statistics[property] for property in properties)))
outf.close()
