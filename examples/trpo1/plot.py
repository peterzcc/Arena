import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

scale = 1e6


def read_data(data_dir, dataname, width=1, batch=1):
    data = pd.read_csv('{}/train_log.csv'.format(data_dir))
    y = data[dataname].rolling(width, center=False, min_periods=width).mean()
    x = (1.0 / batch) * data['t'].values / scale
    print("latest value: {}".format(y.iloc[-1]))
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
    parser.add_argument('--xaxis', nargs='+', help='x axis', type=str, required=False, default="timestep (million)")
    parser.add_argument('--yaxis', nargs='+', help='y axis', type=str, required=False, default="total reward")
    parser.add_argument('--a', '-a', required=False, type=float, default=0.8,
                        help='alpha')

    args = parser.parse_args()
    dirs = ["."] if args.dir == "" else args.dir
    labels = [None] if args.label == "" else args.label
    datas = []
    for data_dir in dirs:
        x, y = read_data(data_dir, args.dataname, args.width, args.batch)
        datas.append([x, y])
    for i, data in enumerate(datas):
        x = data[0]
        y = data[1]
        if args.n is not None:
            x = x[:args.n]
            y = y[:args.n]
        if args.t is not None:
            exceeds = np.flatnonzero(x > args.t)
            max_id = exceeds[0] if len(exceeds) != 0 else len(x)
            x = x[:max_id]
            y = y[:max_id]
        plt.plot(x, y, linewidth=0.5, alpha=args.a)
    if labels[0] is not None:
        plt.legend(labels, loc='best')
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, np.maximum(y1, -1000.), y2))
    plt.xlabel(args.xaxis)
    plt.ylabel(args.yaxis)
    plt.savefig('vis_train' + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
