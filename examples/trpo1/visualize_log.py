import matplotlib.pyplot as plt
import re
import argparse
import os
import sys
import numpy as np
# import mujoco_py as mj
import pandas as pd


def get_loss(dir, mode=None, name="", extra_str="move1", use_epoch=False):
    """
    Plot visualization graph
    Usage: - logfile: log file
       - starting_point: visualize from which point
       - average_interval: The steps we average when computing a single loss for visualization
       - step_interval: The steps we for computing a loss for visualization
    """
    assert dir
    logfile = os.path.realpath('{}/log.txt'.format(dir))
    out_file = sys.stdout
    loss_file = open(logfile, 'r')

    # print("Parsing log from %s" % (logfile))
    loss_list = []

    float_regex = "(?P<target>[+-]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)"
    minor_regex = "(?P<dummy>[+-]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)"
    regex_list = ['Average Return:{0}'.format(float_regex), 'std: {0}'.format(float_regex),
                  'img_loss={0}'.format(float_regex),
                  'act_clips: {0}'.format(float_regex),
                  'new_kl: {0}'.format(float_regex),
                  'Average Return:{0}'.format(float_regex),
                  'ae loss after: {0}'.format(float_regex),
                  'old_mse:{1} \tex_var:{0}'.format(float_regex, minor_regex),
                  'mse:{0}'.format(float_regex),
                  '{1} mean_r_t: {0}'.format(float_regex, extra_str),
                  'ave subt:{0}'.format(float_regex)]
    name_list = ['return', 'std', 'image_loss', 'num_action_overflows', 'kl', 'performance', 'ae_loss', 'expvar',
                 'mse', 'mean_r_t', 'subpolicy_len']
    name_dict = {v:i for i,v in zip(np.arange(len(name_list)),name_list)}
    timestep_list = []
    mode = name_dict[name] if name != "" else mode
    dataname = name_list[mode]
    regex = regex_list[mode]
    current_t = 0
    for line in loss_file:
        m = re.search(regex, line)
        if use_epoch:
            t = re.search('Epoch:([+-]?[0-9]*[.]?[0-9]+)', line)
        else:
            t = re.search('^t: ([+-]?[0-9]*[.]?[0-9]+)', line)
        if t is not None:
            current_t = t.group(1)
            # out_file.write("@" + str(current_t) + "\n")
            # timestep_list.append(num_steps)
        if m is not None:
            loss = m.groupdict()["target"]
            # out_file.write("@" + str(current_t) + "\n")
            # out_file.write(str(loss) + "\n")
            timestep_list.append(current_t)
            loss_list.append(loss)
    t_array = np.array(timestep_list, dtype=np.float32) / (1. if use_epoch else
                                                           1000000)
    loss_list = np.array(loss_list, dtype=np.float32)
    # if mode == 0:
    #     plt.plot(list(range(len(loss_list) - starting_point)), loss_list[starting_point:], '.', markersize=1.0)
    # elif mode == 1:
    #     # plt.plot(1.0 * np.array(list(range(len(loss_list) - starting_point))), loss_list[starting_point:])
    #     l_data = np.minimum(t_array.shape[0], len(loss_list))
    #     plt.plot(t_array[starting_point:l_data], loss_list[starting_point:l_data], '.', markersize=1.0)
    #
    # elif mode == 4:
    #
    #     kls = np.array(np.log10(np.array(loss_list[starting_point:]).astype(
    #         np.float32) + 1e-6))  # loss_list[starting_point:]).astype(np.float32) #
    #     plt.plot(
    #         np.array(list(range(len(loss_list) - starting_point))),
    #         kls,
    #         '.', markersize=1.0)
    #     # x1, x2, y1, y2 = plt.axis()
    #     #
    #     # plt.axis((x1, x2, y1, 0.005))
    #     # x1, x2, y1, y2 = plt.axis()
    #     #
    #     # plt.axis((x1, x2, -5.5, -2.0))
    # elif mode == 5:
    #     l_data = np.minimum(t_array.shape[0], len(loss_list))
    #     plt.plot(t_array[starting_point:l_data], loss_list[starting_point:l_data], '.', markersize=1.0)
    # else:
    #     plt.plot(np.array(list(range(len(loss_list) - starting_point))), loss_list[starting_point:])
    log_scale_names = ["kl"]
    if dataname in log_scale_names:
        ys = np.array(np.log10(np.array(loss_list).astype(np.float32) + 1e-6))
    else:
        ys = loss_list

    # plt.xlabel('t')
    # plt.ylabel(name_list[mode])
    # filename = os.path.basename(logfile)
    # plt.savefig('vis_' + name_list[mode] + '_' + filename + '.pdf', bbox_inches='tight')
    # plt.clf()
    print("latest value: {}".format(ys[-1]))
    return t_array, ys

def main():
    parser = argparse.ArgumentParser(description='Script to visualize content in a log file')
    parser.add_argument('--width', '-w', required=False, type=int, default=1,
                        help='window size')
    parser.add_argument('-l', '--logfile', required=False, type=str, default="log.txt", help='Path of the log file.')
    parser.add_argument('-o', '--outpath', required=False, type=str, default="",
                        help='Path of output file.')
    parser.add_argument('--dataname', required=False, type=str, default="",
                        help='Path of output file.')
    parser.add_argument('-e', '--e', required=False, type=int, default=0,
                        help='use epoch')
    parser.add_argument('-m', '--mode', required=False, type=int, default=None,
                        help='mode')
    parser.add_argument('--t', '-t', required=False, type=float, default=None,
                        help='time')
    parser.add_argument('--dir', nargs='+', help='<Required> Set flag', type=str, required=False, default="")
    parser.add_argument('--label', nargs='+', help='<Required> Set flag', type=str, required=False, default="")
    parser.add_argument('--extra', nargs='+', help='<Required> Set flag', type=str, required=False, default="")
    parser.add_argument('--shape', required=False, type=str, default="line", help='dot or line')
    parser.add_argument('--xaxis', help='x axis', type=str, required=False, default="timestep (million)")
    parser.add_argument('--yaxis', help='y axis', type=str, required=False, default="")
    parser.add_argument('--a', '-a', required=False, type=float, default=0.8,
                        help='alpha')
    args, unknown = parser.parse_known_args()
    dirs = ["."] if args.dir == "" else args.dir
    labels = [None] if args.label == "" else args.label
    extra_names = [None] if args.extra == "" else args.extra
    if len(dirs) == 1 and len(extra_names) != 1:
        dirs = (len(extra_names)) * dirs
    elif len(dirs) != 1 and len(extra_names) == 1:
        extra_names = extra_names * len(dirs)
    for data_dir, extra_name in zip(dirs, extra_names):
        x, y = get_loss(data_dir, args.mode, args.dataname, extra_str=extra_name, use_epoch=args.e)
        if args.width != 1:
            y = pd.Series(y).rolling(args.width, center=False, min_periods=args.width).mean()
        if args.t is not None:
            exceeds = np.flatnonzero(x > args.t)
            max_id = exceeds[0] if len(exceeds) != 0 else len(x)
            x = x[:max_id]
            y = y[:max_id]
        if args.shape == "line":
            plt.plot(x, y, linewidth=0.5, alpha=args.a)
        else:
            plt.plot(x, y, '.', markersize=1.0)
    if labels[0] is not None:
        plt.legend(labels, loc='best')
    plt.xlabel(args.xaxis)
    if args.yaxis == "":
        yaxis = args.dataname
    else:
        yaxis = args.yaxis
    plt.ylabel(yaxis)
    plt.savefig('vis_{}.pdf'.format(args.dataname), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()
