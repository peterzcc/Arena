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
import cv2
import argparse


def draw_res(im, roi, delay=0):
    im = im.transpose(1, 2, 0)
    width = im.shape[0]
    height = im.shape[1]
    center = numpy.array([width, height])/2 - roi[0:2] * numpy.array([width, height])
    center = numpy.int64(center)
    h = numpy.int64(height * roi[2])
    im2 = numpy.zeros(im.shape)
    im2[:] = im
    cv2.circle(im2, (center[0], center[1]), 4, (0, 0, 255))
    cv2.line(im2, (1, center[1] - h/2), (width-1, center[1] - h/2), (0, 0, 255))
    cv2.line(im2, (1, center[1] + h/2), (width-1, center[1] + h/2), (0, 0, 255))
    cv2.imshow('image', im2[:,:,::-1]/255.0)
    cv2.waitKey(delay)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
# ch = logging.FileHandler('test_vrep_game.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

parser = argparse.ArgumentParser('script to test vrep game wrapper')
parser.add_argument('--port', default=19997, type=int, help='remote port on the vrep server side')
args, unknown = parser.parse_known_args()

rng = get_numpy_rng()
itr = 1000000
game = VREPGame(replay_start_size=100, remote_port=args.port)

#test basic functions of the V-REP simulator
start = time.time()
count = 1
game.begin_episode()
for i in xrange(itr):
    if game.episode_terminate():
        end = time.time()
        logging.info('Episode:%d, Reward:%f, fps:%f' % (count, game.episode_reward, game.episode_step / (end-start)))
        game.begin_episode()
        start = time.time()
        count += 1
    a = rng.normal(0, 0.1, (4, ))
    game.play(a)
    if i%1 == 0:
        # game.print_vrep_data()
        roi = game.target_coordinates
        im = game.image
        draw_res(im, roi, delay=10)



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


