class Game(object):

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
        raise NotImplementedError

    def current_state(self):
        raise NotImplementedError

    def play(self, a):
        raise NotImplementedError