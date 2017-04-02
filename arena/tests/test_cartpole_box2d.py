from arena.games.cartpole_box2d import CartpoleSwingupEnv
import numpy as np

env = CartpoleSwingupEnv()
n_itr = 100
T = 500

for itr in xrange(n_itr):
    observations = []
    actions = []
    rewards = []

    observation = env.reset()
    for step in xrange(T):
        action = np.random.randn(1)
        next_observation, reward, terminal, _ = env.step(action)
        env.render()
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        observation = next_observation
        if terminal:
            break

    print 'Accumulate reward is %f' % (np.sum(rewards))