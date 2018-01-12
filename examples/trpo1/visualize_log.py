import matplotlib.pyplot as plt
import re
import argparse
import os
import sys
import numpy as np
import mujoco_py as mj

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
    regex_list = ['Average Return:([+-]?[0-9]*[.]?[0-9]+)', ' std: ([+-]?[0-9]*[.]?[0-9]+)',
                  'img_loss=([+-]?[0-9]*[.]?[0-9]+)',
                  'act_clips: ([+-]?[0-9]*[.]?[0-9]+)',
                  'new kl: ([+-]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)',
                  'Average Return:([+-]?[0-9]*[.]?[0-9]+)',
                  'ae loss after: ([+-]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)',
                  'ae expvar: ([+-]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)']
    name_list = ['Return', 'Std', 'image_loss', 'Num. action overflows', 'KL', 'Performance', 'ae loss', 'ae expvar']
    timestep_list = []
    for line in loss_file:
        m = re.search(regex_list[mode], line)
        t = re.search('^t: ([+-]?[0-9]*[.]?[0-9]+)', line)
        if m is not None:
            loss = m.group(1)
            out_file.write(str(loss) + "\n")
            loss_list.append(loss)
        if t is not None:
            num_steps = t.group(1)
            out_file.write("@" + str(num_steps) + "\n")
            timestep_list.append(num_steps)
    t_array = np.array(timestep_list, dtype=np.float32) / 1000000
    loss_list = np.array(loss_list, dtype=np.float32)
    if mode == 0:
        plt.plot(list(range(len(loss_list) - starting_point)), loss_list[starting_point:], '.', markersize=1.0)
    elif mode == 1:
        # plt.plot(1.0 * np.array(list(range(len(loss_list) - starting_point))), loss_list[starting_point:])
        l_data = np.minimum(t_array.shape[0], len(loss_list))
        plt.plot(t_array[starting_point:l_data], loss_list[starting_point:l_data], '.', markersize=1.0)
    elif mode == 4:
        kls = np.log10(np.array(loss_list[starting_point:]).astype(np.float32) + 1e-6)
        plt.plot(
            np.array(list(range(len(loss_list) - starting_point))),
            kls,
            '.')
    elif mode == 5:
        l_data = np.minimum(t_array.shape[0], len(loss_list))
        plt.plot(t_array[starting_point:l_data], loss_list[starting_point:l_data], '.', markersize=1.0)
    else:
        plt.plot(np.array(list(range(len(loss_list) - starting_point))), loss_list[starting_point:])
    plt.xlabel('t')
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
