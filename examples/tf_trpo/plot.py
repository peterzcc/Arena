import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='Plot.')
    parser.add_argument('--width', '-w', required=False, type=int, default=500,
                        help='batch size')
    args = parser.parse_args()
    data = pd.read_csv('train_log.csv')
    data["rs"] = data['Reward'].rolling(args.width, center=False, min_periods=0).mean()
    plt.plot(data['t'].values, data["rs"].values)
    plt.show()


if __name__ == '__main__':
    main()
