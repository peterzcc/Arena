import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

scale = 1e6
STATS_KEYS = ['mean', 'logstd', 'advantage', 'baseline', 'return', "old_logpi"]


def read_data(data_dir, dataname, width=1, batch=1):
    data = pd.read_csv('{}/train_log.csv'.format(data_dir))
    y = data[dataname].rolling(width, center=False, min_periods=width).mean()
    x = (1.0 / batch) * data['t'].values / scale
    return x, y


def main():
    parser = argparse.ArgumentParser(description='Plot.')
    parser.add_argument('--width', '-w', required=False, type=int, default=20,
                        help='window size')
    parser.add_argument('--batch', '-b', required=False, type=int, default=1,
                        help='batch size')
    parser.add_argument('--x', '-x', required=False, type=int, default=None,
                        help='x axis')

    parser.add_argument('--n', '-n', required=False, type=int, default=None,
                        help='length')
    parser.add_argument('--t', '-t', required=False, type=float, default=None,
                        help='time')
    parser.add_argument('--dataname', default="Reward", type=str, help='name')
    parser.add_argument('--dir', nargs='+', help='<Required> Set flag', type=str, required=False, default="")
    parser.add_argument('--label', nargs='+', help='<Required> Set flag', type=str, required=False, default="")
    args = parser.parse_args()
    # dir = args.dir
    dir = "."
    stats_path = "{}/stats.h5".format(dir)
    epoch = 0
    batch_size = 2048
    start = epoch * batch_size
    end = (epoch + 1) * batch_size
    stats = pd.HDFStore(stats_path, 'r')
    envname = "flatmove1d"
    paths = {}
    for k in STATS_KEYS:
        paths[k] = stats.select("{}/{}".format(envname, k), start=start, stop=end).values

    # df_stats = ({k: pd.DataFrame(paths[k]) for k in STATS_KEYS})
    # old_logpi = batch_run_forward(self.policy.old_likelihood, feed=feed, N=times.size, session=self.session)
    # df_stats["old_logpi"] = pd.DataFrame(old_logpi)

    # dirs = ["."] if args.dir == "" else args.dir
    # labels = [None] if args.label == "" else args.label
    # datas = []
    # for data_dir in dirs:
    #     x, y = read_data(data_dir, args.dataname, args.width, args.batch)
    #     datas.append([x, y])
    # for i, data in enumerate(datas):
    #     x = data[0]
    #     y = data[1]
    #     if args.n is not None:
    #         x = x[:args.n]
    #         y = y[:args.n]
    #     if args.t is not None:
    #         exceeds = np.flatnonzero(x > args.t)
    #         max_id = exceeds[0] if len(exceeds) != 0 else len(x)
    #         x = x[:max_id]
    #         y = y[:max_id]
    #     plt.plot(x, y, linewidth=0.5)
    # if labels[0] is not None:
    #     plt.legend(labels, loc='best')
    plt.plot(x, y, '.', markersize=1.0)
    plt.savefig('vis_stats' + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
