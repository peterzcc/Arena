from arena.agents import Agent
import mxnet as mx
#TODO: implement training
class DqnAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0,
                 f_get_sym = None,
                 replay_memory_size=1000000,
                 replay_start_size=100,
                 history_length=4,
                 minibatch_size = 32,
                 optimizer=mx.optimizer.create(name='adagrad', learning_rate=0.01, eps=0.01),
                 ctx=mx.cpu()
                 ):
        super(DqnAgent, self).__init__(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid
        )

    def act(self, observation):
        raise NotImplementedError

    def receive_feedback(self, reward, done):
        raise NotImplementedError