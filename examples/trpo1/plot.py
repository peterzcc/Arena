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
    args = parser.parse_args()
    data = pd.read_csv('train_log.csv')
    data["rs"] = data['Reward'].rolling(args.width, center=False, min_periods=0).mean()
    x = 1.0 / args.batch * data['t'].values
    plt.plot(x, data["rs"].values)
    plt.savefig('vis_train' + '.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()