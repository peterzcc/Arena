class Model(object):
    def __init__(self, observation_space, action_space):
        pass

    def predict(self, observation):
        raise NotImplementedError

    def compute_update(self, train_dataa):
        raise NotImplementedError

    def update(self, diff, new=None):
        raise NotImplementedError


class ModelWithCritic(object):
    def __init__(self, observation_space, action_space):
        pass

    def predict(self, observation):
        raise NotImplementedError

    def compute_update(self, train_dataa):
        raise NotImplementedError

    def compute_critic(self, states):
        raise NotImplementedError

    def update(self, diff, new=None):
        raise NotImplementedError
