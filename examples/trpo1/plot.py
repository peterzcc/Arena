import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot.')
    parser.add_argument('--width', '-w', required=False, type=int, default=500,
                        help='window size')
    parser.add_argument('--batch', '-b', required=False, type=int, default=1,
                        help='batch size')
    parser.add_argument('--x', '-x', required=False, type=int, default=None,
                        help='x axis')

    parser.add_argument('--n', '-n', required=False, type=int, default=None,
                        help='length')
    parser.add_argument('--dataname', default="Reward", type=str, help='name')
    args = parser.parse_args()
    data = pd.read_csv('train_log.csv')
    data["rs"] = data[args.dataname].rolling(args.width, center=False, min_periods=args.width).mean()
    x = (1.0 / args.batch) * data['t'].values / 1000
    if args.n is not None:
        plt.plot(x[:args.n], data["rs"].values[:args.n], linewidth=0.5)
    else:
        plt.plot(x / 1000, data["rs"].values, linewidth=0.5)
    plt.savefig('vis_train' + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
