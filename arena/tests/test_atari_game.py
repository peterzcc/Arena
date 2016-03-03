from ..games import AtariGame
import numpy
import time
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

replay_start_size = 1000
game = AtariGame(resize_mode='scale', replay_start_size=replay_start_size, display_screen=True)
game.begin_episode()
start = time.time()
totoal_time_step = 10000
minibatch_size = 32
sample_total_time = 0
sample_total_num = 0
plt.figure()


for i in xrange(totoal_time_step):
    action = numpy.random.randint(0, len(game.action_set))
    reward, terminate_flag = game.play(action)
    if game.episode_terminate:
        print game.episode_reward, game.episode_step
        game.begin_episode()
    if game.state_enabled:
        current_state = game.current_state()
    if game.replay_memory.sample_enabled:
        sample_start = time.time()
        states, actions, rewards, next_states, terminate_flags = game.replay_memory.sample(minibatch_size)
        gpu_states = nd.array(states, ctx=mx.gpu())/float(255.0)
        gpu_next_states = nd.array(next_states, ctx=mx.gpu())/float(255.0)
        #plt.imshow(gpu_states.asnumpy()[0,0,:,:] * 255, cmap=cm.Greys_r)
        #plt.show()
        sample_end = time.time()
        sample_total_time += sample_end - sample_start
        sample_total_num += 1
end = time.time()
print totoal_time_step/float(end - start)
print sample_total_num / sample_total_time