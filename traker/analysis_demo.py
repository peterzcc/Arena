import re
import argparse
import sys
parser = argparse.ArgumentParser(description='Demo script to aggregate training statistics from log file')
parser.add_argument('-f', '--file', required=True, type=str,
                    help = 'path of log file')
parser.add_argument('-o', '--output', default=None, type=str,
                    help = 'path of output file')
args, unknown = parser.parse_known_args()
f = open(args.file, 'r')

training_statistics = {}
properties = ['Validation-accuracy', 'Train-accuracy', 'Time cost']
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

outf = open(args.output, 'w') if args.output is not None else sys.stdout
for epoch_id, statistics in training_statistics.items():
    outf.write("%d %s\n" %(epoch_id, ' '.join(statistics[property] for property in properties)))
outf.close()