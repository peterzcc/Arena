import numpy as np
import scipy


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# TODO: add agent info,important
class AcMemory(object):
    def __init__(self, observation_shape, action_shape, max_size, gamma=0.995, use_gae=False,
                 lam=0.96, get_critic_online=True, info_shape=None):
        self.observation_buffer = np.empty(shape=(max_size,) + observation_shape,
                                           dtype=np.float32)
        self.action_buffer = np.empty(shape=(max_size,) + action_shape, dtype=np.float32)
        self.reward_buffer = np.empty(shape=(max_size,), dtype=np.float32)
        self.q_buffer = np.empty(shape=(max_size,), dtype=np.float32)
        self.adv_buffer = np.empty(shape=(max_size,), dtype=np.float32)
        self.critic_buffer = np.empty(shape=(max_size,), dtype=np.float32)
        self.t_buffer = np.empty(shape=(max_size,), dtype=np.float32)
        self.Tmax = 0
        self.t0 = 0
        self.num_episodes = 0
        self.gamma = gamma
        self.lam = lam
        self.episode_start_times = []
        self.info_buffer = {}


        if get_critic_online:
            self.append_state = self.append_state_with_critic
        else:
            self.append_state = self.append_state_without_critic
        if info_shape is not None:
            for k, shape in info_shape.items():
                self.info_buffer[k] = np.empty(shape=(max_size,) + shape, dtype=np.float32)

        if not use_gae:
            self.add_path = self.add_td_path
            self.extract_all = self.extract_all_without_normalize
        else:
            self.add_path = self.add_gae_path
            self.extract_all = self.extract_all_with_normalize

    def append_state_without_critic(self, observation, action, info=dict(), critic=None):
        self.Tmax += 1
        last_idx = self.Tmax - 1
        self.observation_buffer[last_idx] = observation
        self.action_buffer[last_idx] = action
        self.t_buffer[last_idx] = self.Tmax - self.t0
        for k, v in info.items():
            self.info_buffer[k][last_idx] = v

    def append_state_with_critic(self, observation, action, info=dict(), critic=None):
        self.append_state_without_critic(observation, action, info=info)
        self.critic_buffer[self.Tmax - 1] = critic

    def fill_episode_critic(self, f_get_critic):
        self.critic_buffer[self.t0:self.Tmax] = \
            f_get_critic({"observations": self.observation_buffer[self.t0:self.Tmax],
                          "times": self.q_buffer[self.t0:self.Tmax]
                          })

    def append_feedback(self, reward):
        self.reward_buffer[self.Tmax - 1] = reward

    def add_td_path(self, done=True):
        assert done is True
        rewards = self.reward_buffer[self.t0:self.Tmax]
        self.q_buffer[self.t0:self.Tmax] = \
            discount(rewards, self.gamma)
        self.adv_buffer[self.t0:self.Tmax] = \
            self.q_buffer[self.t0:self.Tmax] - \
            self.critic_buffer[self.t0:self.Tmax]
        self.t0 = self.Tmax

        self.episode_start_times.append(self.t0)
        self.t0 = self.Tmax

    def add_gae_path(self, done):
        rewards = self.reward_buffer[self.t0:self.Tmax]
        self.q_buffer[self.t0:self.Tmax] = \
            discount(rewards, self.gamma)
        critics = self.critic_buffer[self.t0:self.Tmax]
        end_value = 0.0 if done else critics[-1]
        deltas = np.empty_like(rewards)
        # print("eps d:", deltas.size,self.t0,self.Tsum)
        deltas[:-1] = rewards[:-1] + self.gamma * critics[1:] - end_value
        deltas[-1] = rewards[-1] + self.gamma * end_value - end_value
        self.adv_buffer[self.t0:self.Tmax] = \
            discount(deltas, self.gamma * self.lam)

        self.episode_start_times.append(self.t0)
        self.t0 = self.Tmax

    def reset(self):
        self.t0 = 0
        self.Tmax = 0
        self.episode_start_times.clear()

    def extract_all_without_normalize(self):
        result = dict(
            observations=self.observation_buffer[:self.Tmax],
            times=self.t_buffer[:self.Tmax],
            actions=self.action_buffer[:self.Tmax],
            values=self.q_buffer[:self.Tmax],
            advantages=self.adv_buffer[:self.Tmax],
            agent_infos={k: v[:self.Tmax] for k, v in self.info_buffer.items()}
        )
        return result

    def extract_all_with_normalize(self):
        alladv = self.adv_buffer[:self.Tmax]
        std = alladv.std()
        mean = alladv.mean()
        self.adv_buffer[:self.Tmax] -= mean
        self.adv_buffer[:self.Tmax] /= std

        return self.extract_all_without_normalize()
