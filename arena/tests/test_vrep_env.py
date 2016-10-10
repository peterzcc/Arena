import numpy
import logging
import time
import sys

from arena.games.vrep_gym import VREPHierarchyGame

logger = logging.getLogger()
# ch = logging.StreamHandler(sys.stdout)
ch = logging.FileHandler('a.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

game = VREPHierarchyGame(headless=True, action_type='discrete', server_silent=True)

for itr in range(10):
    observation = game.reset()
    start_time = time.time()
    rewards = []
    for t in range(100):
        state = game.current_state()
        print 'observation:'
        print observation
        print 'current state:'
        print state
        # action = numpy.random.normal(0, 0.1, (4,))
        action = game.env.action_space.sample()
        observation, reward, done, info = game.step(action)
        rewards.append(reward)
        if done:
            break
    end_time = time.time()
    print "Episode %d finished after %d timesteps, Rewards:%f" % (itr, t + 1, numpy.sum(rewards))
    print 'replay memory size:', game.replay_memory.size
    if game.replay_memory.sample_enabled:
        states, actions, rewards, next_states, terminate_flags \
            = game.sample(32)
        print 'replay memory content:'
        for i in xrange(game.replay_memory.size):
            print i
            print game.replay_memory.states[i]
        print 'sampled content'
        for i in xrange(32):
            print states[i]
            print next_states[i]
