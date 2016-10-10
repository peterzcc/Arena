import gym


class Agent(object):
    def __init__(self, observation_space, action_space):
        """

        Parameters
        ----------
        observation_space : gym.Space
        action_space : gym.Space
        """
        raise NotImplementedError

    def act(self, observation, is_learning=False):
        """

        Parameters
        ----------
        observation : gym.Space
        reward : float
        done : bool
        is_learning : bool

        Returns
        -------

        """
        raise NotImplementedError

    def receive_feedback(self, reward, done):
        raise NotImplementedError

    def save_parameters(self, path):
        # TODO: implement
        raise NotImplementedError

    def load_parameters(self, path):
        # TODO: implement
        raise NotImplementedError

    def stats_keys(self):
        return []

    def stats_values(self):
        return []


class RandomAgent(Agent):
    def __init__(self, observation_space, action_space):
        """

        Parameters
        ----------
        observation_space : gym.Space
        action_space : gym.Space
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation, is_learning=False):
        """

        Parameters
        ----------
        observation : gym.Space
        reward : float
        done : bool
        is_learning : bool

        Returns
        -------
        """

        return self.action_space.sample()

    def receive_feedback(self, reward, done):
        pass

    def save_parameters(self, path):
        pass

    def load_parameters(self, path):
        pass


