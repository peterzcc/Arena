import numpy as np
import os.path as osp
import pygame

from arena.games.box2d.xml_box2d import find_body
from arena.games.box2d.box2d_env import Box2DEnv


# Tornio, Matti, and Tapani Raiko. "Variational Bayesian approach for
# nonlinear identification and control." Proc. of the IFAC Workshop on
# Nonlinear Model Predictive Control for Fast Systems, NMPC FS06. 2006.
class CartpoleSwingupEnv(Box2DEnv):

    def __init__(self, *args, **kwargs):
        model_path = osp.abspath(osp.join(osp.dirname(__file__),'box2d', 'models', 'cartpole.xml.mako'))
        super(CartpoleSwingupEnv, self).__init__(model_path=model_path, *args, **kwargs)
        self.max_cart_pos = 3
        self.max_reward_cart_pos = 3
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")

    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        bounds = np.array([
            [-1, -2, np.pi-1, -3],
            [1, 2, np.pi+1, 3],
        ])
        low, high = bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_current_obs()

    def compute_reward(self, action):
        yield
        if self.is_current_done():
            yield -100
        else:
            if abs(self.cart.position[0]) > self.max_reward_cart_pos:
                yield -1
            else:
                yield np.cos(self.pole.angle)

    def is_current_done(self):
        return abs(self.cart.position[0]) > self.max_cart_pos

    def action_from_keys(self, keys):
        if keys[pygame.K_LEFT]:
            return np.asarray([-10])
        elif keys[pygame.K_RIGHT]:
            return np.asarray([+10])
        else:
            return np.asarray([0])

