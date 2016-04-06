__author__ = 'flyers'

from arena.games import VREPGame
from arena.utils import *
import numpy
import time
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt

rng = get_numpy_rng()
game = VREPGame(replay_start_size=100)

# test simulator environment
game.begin_episode()
for i in xrange(500):
    if game.episode_terminate():
        game.begin_episode()
    a = rng.normal(5.335, 0.1, (4, ))
    game.play(a)
    if i%1 == 0:
        game.print_vrep_data()
        # plt.imshow(game.image)
        # plt.show()


for i in xrange(10):
    states, actions, next_states, rewards, terminate_flag = game.replay_memory.sample(5)
    print states, actions, next_states, rewards

ch = raw_input("Press Any Key to Continue")
game.begin_episode()
total_time_step = 1000
minibatch_size = 32
sample_total_time = 0
sample_total_num = 0

start = time.time()
for i in xrange(total_time_step):
    if game.episode_terminate():
        print game.episode_step, game.episode_reward
        game.begin_episode()
    a = rng.normal(5.335, 0.1, (4, ))
    reward, terminate_flag = game.play(a)
    print 'single step reward is %.4f' % reward


end = time.time()
print total_time_step/float(end-start)


