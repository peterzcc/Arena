
DEFAULT_MAX_EPISODE_STEP = 1000000

class Game(object):
    def __init__(self):
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_step = 0
        self.max_episode_step = DEFAULT_MAX_EPISODE_STEP
        self.replay_memory = None

    def start(self):
        raise NotImplementedError("Must Implement!")

    def begin_episode(self, max_episode_step):
        raise NotImplementedError("Must Implement!")

    @property
    def episode_terminate(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    @property
    def state_enabled(self):
        return self.replay_memory.size >= self.replay_memory.history_length

    def current_state(self):
        return self.replay_memory.latest_slice()

    def play(self, a):
        raise NotImplementedError