from arena.agents import Agent
import multiprocessing as mp
import cv2
from matplotlib import pyplot as plt
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
        cv2.imshow("observation", observation[1])
        cv2.waitKey(30)
        print(observation[0])

        # plt.figure(1)
        # plt.imshow(observation[1], cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.draw()
        # plt.pause(0.001)

        return self.action_space.sample()

    def receive_feedback(self, reward, done, info={}):
        pass
        # if self.is_learning.value:
        #     if self.lc_t % 4 == 0:
        #         self.params += 1
