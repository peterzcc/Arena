from arena.games import CartPoleGame
import numpy
import matplotlib.pyplot as plt
import time

replay_start_size = 1000
game = CartPoleGame(mode='swingup', pole_scales=numpy.array([1.]), noise=1.0, reward_noise=1.0,
                    replay_start_size=replay_start_size, display_screen=False)
game.start()
plt.ion()
plt.show()
for i in range(10000):
    reward, is_terminate = game.play(numpy.random.randint(2))
    game.draw()
    if game.state_enabled:
        obs = game.get_observation()
        print obs
    if game.episode_terminate:
        game.start()
    if game.replay_memory.sample_enabled:
        game.replay_memory.sample(32)
    print reward, is_terminate
    print game.get_observation()
    time.sleep(0.1)
plt.show()

