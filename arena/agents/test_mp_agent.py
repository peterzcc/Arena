from arena.agents import Agent
import multiprocessing as mp


class TestMpAgent(Agent):
    def __init__(self, observation_space, action_space,
                 shared_params, stats_rx, acts_tx,
                 is_learning, global_t, pid=0, **kwargs):
        super(TestMpAgent, self).__init__(
            observation_space, action_space,
            shared_params, stats_rx, acts_tx,
            is_learning, global_t, pid, **kwargs
        )

    def act(self, observation):
        return self.action_space.sample()

    def receive_feedback(self, reward, done):
        if self.is_learning.value:
            if self.lc_t % 4 == 0:
                self.params += 1
