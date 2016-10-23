import numpy as np


class EpsGreedy(object):
    def __init__(self,
                 eps_0,
                 eps_t,
                 t_max, p_assign):
        if not isinstance(eps_0,float):
            if eps_0.shape != eps_t.shape or eps_0.shape != p_assign:
                raise ValueError("incompatible parameter size")
        self.eps_t = eps_t
        self.eps_decay = (eps_0-eps_t)/t_max
        self.eps_id = 0
        self.ps = p_assign
        self.all_eps_current = eps_0

    def update_t(self, t):
        self.all_eps_current = \
            np.maximum(self.all_eps_current-t*self.eps_decay, self.eps_t)

    def update_id(self):
        self.eps_id = np.random.choice(self.ps.shape[0], p=self.ps)

    def decide_exploration(self):
        do_exploration = np.random.rand() < self.all_eps_current[self.eps_id]
        return do_exploration

