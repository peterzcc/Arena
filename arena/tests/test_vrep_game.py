__author__ = 'flyers'

from arena.games import VREPGame
from arena.utils import *
import numpy
import time
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import logging
import sys
from arena.helpers.visualization import draw_track_res

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch = logging.FileHandler('test_vrep_game.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

rng = get_numpy_rng()
game = VREPGame(replay_start_size=100)

#test basic functions of the V-REP simulator
game.begin_episode()
for i in xrange(500):
    if game.episode_terminate():
        game.begin_episode()
    a = rng.normal(5.335, 0.1, (4, ))
    game.play(a)
    if i%1 == 0:
        # game.print_vrep_data()
        # plt.imshow(game.image)
        # plt.show()
        roi = game.target_coordinates
        im = game.image.transpose(2, 0, 1)
        roi /= 128
        draw_track_res(im, roi, delay=10)


# test fps of V-REP
game = VREPGame(replay_start_size=100, replay_memory_size=500,
                history_length=5)
N = 1
T = 100
n_itr = 100
discount = 0.99

start = time.time()
for itr in xrange(n_itr):
    for i in xrange(N):
        game.begin_episode()
        observations = []
        actions = []
        rewards = []
        for t in xrange(T):
            observation = game.current_state()
            action = rng.normal(0, 0.5, (4, ))
            reward, terminate_flag = game.play(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            if terminate_flag:
                break

        observations = numpy.array(observations)
        actions = numpy.array(actions)
        rewards = numpy.array(rewards)
        batch_size = observations.shape[0]
        logging.info('Batchsize:%d, Accumulate Reward:%f' % (batch_size, rewards.sum()))
        logging.info('Observations:')
        logging.info(observations)
        logging.info('Actions:')
        logging.info(actions)
        logging.info('Rewards:')
        logging.info(rewards)
        print observations.shape


end = time.time()


