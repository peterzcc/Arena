from ..games import AtariGame
import numpy
import time
import mxnet as mx
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

replay_start_size = 1000
game = AtariGame(resize_mode='scale', replay_start_size=replay_start_size, display_screen=False)

game.begin_episode()
for i in range(2000):
    if game.episode_terminate:
        game.begin_episode()
    game.play(numpy.random.randint(len(game.action_set)))
new_replay_memory = game.replay_memory.copy()
print new_replay_memory.size, new_replay_memory.top
print id(new_replay_memory.states), id(game.replay_memory.states), id(new_replay_memory.size), id(new_replay_memory.top)
for i in xrange(10):
    states, actions, rewards, next_states, terminate_flags = new_replay_memory.sample(32)
    gpu_states = nd.array(states, ctx=mx.gpu())/float(255.0)
    gpu_next_states = nd.array(next_states, ctx=mx.gpu())/float(255.0)
#    plt.imshow(gpu_states.asnumpy()[0,0,:,:] * 255, cmap=cm.Greys_r)
#    plt.show()
ch = raw_input("Press Any Key to Continue")
game.begin_episode()
totoal_time_step = 10000
minibatch_size = 32
sample_total_time = 0
sample_total_num = 0
plt.figure()

start = time.time()
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