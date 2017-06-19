import matplotlib.pyplot as plt
import re
import argparse
import os
import sys
import numpy as np


def get_loss(logfile, outpath, starting_point=1, mode=0):
    """
    Plot visualization graph
    Usage: - logfile: log file
       - starting_point: visualize from which point
       - average_interval: The steps we average when computing a single loss for visualization
       - step_interval: The steps we for computing a loss for visualization
    """
    assert logfile
    logfile = os.path.realpath(logfile)
    if outpath == "":
        out_file = sys.stdout
    else:
        out_file = open(outpath, 'w')
    loss_file = open(logfile, 'r')

    print("Parsing log from %s" % (logfile))
    loss_list = []
    regex_list = ['Average Return:([+-]?[0-9]*[.]?[0-9]+)', ' std: ([+-]?[0-9]*[.]?[0-9]+)']
    name_list = ['Return', 'Std']
    for line in loss_file:
        m = re.search(regex_list[mode], line)
        if m is not None:
            loss = m.group(1)
            out_file.write(str(loss) + "\n")
            loss_list.append(loss)
    if mode == 0:
        plt.plot(list(range(len(loss_list) - starting_point)), loss_list[starting_point:], '.')
    elif mode == 1:
        plt.plot(2 * np.array(list(range(len(loss_list) - starting_point))), loss_list[starting_point:])
    plt.xlabel('Epoch')
    plt.ylabel(name_list[mode])
    filename = os.path.basename(logfile)
    plt.savefig('vis_' + name_list[mode] + '_' + filename + '.pdf', bbox_inches='tight')
    plt.clf()

def main():
    parser = argparse.ArgumentParser(description='Script to visualize content in a log file')
    parser.add_argument('-l', '--logfile', required=False, type=str, default="log.txt", help='Path of the log file.')
    parser.add_argument('-o', '--outpath', required=False, type=str, default="",
                        help='Path of output file.')
    parser.add_argument('-s', '--starting_point', required=False, type=int, default=1,
                        help='Starting point of curve.')
    parser.add_argument('-m', '--mode', required=False, type=int, default=0,
                        help='mode')
    args, unknown = parser.parse_known_args()
    get_loss(args.logfile, args.outpath, args.starting_point, args.mode)

if __name__ == '__main__':
    main()
