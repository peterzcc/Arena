import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
scale = 1e6
STATS_KEYS = ['mean', 'logstd', 'advantage', 'baseline', 'return', "old_logpi"]  # action
dir = "."
stats_path = "{}/stats.h5".format(dir)

epoch = 0


# def plot_dist(epoch):
#     batch_size = 2048
#     start = epoch * batch_size
#     end = (epoch+1) * batch_size
#     stats = pd.HDFStore(stats_path, 'r')
#     envname = "flatmove1d"
#     paths = {}
#     for k in STATS_KEYS:
#         paths[k] = stats.select("{}/{}".format(envname, k), start=start, stop=end).values
#     oldpi = (np.sum(paths['logstd'], axis=-1)+paths['old_logpi'] +
#              0.5 * paths['logstd'].shape[-1] * np.log(2 * np.pi)) *(np.exp(np.sum(paths['logstd']))+1e-6)**-2
#     y = paths['return'] - paths['baseline']
#     plt.plot(oldpi, y, '.', markersize=1.0)
#     plt.savefig('vis_stats_{}.pdf'.format(epoch), bbox_inches='tight')
#     plt.clf()

def plot_dist(epoch):
    batch_size = 2048
    start = epoch * batch_size
    end = (epoch + 1) * batch_size
    stats = pd.HDFStore(stats_path, 'r')
    envname = "flatmove1d"
    paths = {}
    for k in STATS_KEYS:
        paths[k] = stats.select("{}/{}".format(envname, k), start=start, stop=end).values
    logstdsum_value = np.sum(paths['logstd'][0, :])
    # sum_logstd = logstdsum_value * np.ones_like(paths['logstd'])
    # exp_std_pow2 = (np.exp(sum_logstd)+1e-6)**-2
    oldpi = (logstdsum_value + paths['old_logpi'] +
             0.5 * paths['logstd'].shape[-1] * np.log(2 * np.pi))  # *(np.exp(logstdsum_value))**-2
    y = paths['return'] - paths['baseline']
    plt.plot(oldpi, y, '.', markersize=1.0)
    plt.savefig('vis_stats_{}.pdf'.format(epoch), bbox_inches='tight')
    plt.clf()


def ave_adv(epoch):
    batch_size = 2048
    start = epoch * batch_size
    end = (epoch + 1) * batch_size
    stats = pd.HDFStore(stats_path, 'r')
    envname = "flatmove1d"
    paths = {}
    for k in STATS_KEYS:
        paths[k] = stats.select("{}/{}".format(envname, k), start=start, stop=end).values
    logstdsum_value = np.sum(paths['logstd'][0, :])
    oldpi = (logstdsum_value + paths['old_logpi'] +
             0.5 * paths['logstd'].shape[-1] * np.log(2 * np.pi))
    y = paths['return'] - paths['baseline']
    plt.plot(oldpi, y, '.', markersize=1.0)
    plt.savefig('vis_stats_{}.pdf'.format(epoch), bbox_inches='tight')
    plt.clf()


plot_dist(epoch)
