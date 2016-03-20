import numpy
from arena import ReplayMemory
import math
from .game import Game
from .game import DEFAULT_MAX_EPISODE_STEP
from arena.utils import *


class CartPoleGame(Game):
    """Cart Pole environment. This implementation allows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """
    name = "Cart Pole"

    def __init__(self, mode='easy', pole_scales=numpy.array([1.]), noise=0.0, reward_noise=0.0,
                 replay_memory_size=1000000, replay_start_size=100,
                 random_start=True, display_screen=False):
        super(CartPoleGame, self).__init__()
        self.npy_rng = get_numpy_rng()
        self.noise = noise
        self.reward_noise = reward_noise
        self.random_start = random_start
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = numpy.zeros((len(pole_scales),), dtype='float32')
        self.pole_velocity = numpy.zeros((len(pole_scales),), dtype='float32')

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        pole_scales = numpy.array(pole_scales)
        self.pole_length = numpy.ones((len(pole_scales), ))*0.5 * pole_scales
        self.pole_mass = numpy.ones((len(pole_scales), ))*0.1 * pole_scales

        self.domain_name = "Cart Pole"

        self.mode = mode
        if mode == 'hard':
            self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
                                            [-5., 5.],                                    # Cart velocity bound
                                            [-numpy.pi * 45./180., numpy.pi * 45./180.], # Pole angle bounds
                                            [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
            self.mu_c = 0.0005
            self.mu_p = 0.000002
            self.sim_steps = 10
            self.discount_factor = 0.999
        elif mode == 'swingup':
            self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
                                            [-5., 5.],                                   # Cart velocity bound
                                            [-numpy.pi, numpy.pi],                       # Pole angle bounds
                                            [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
            self.mu_c = 0.0005
            self.mu_p = 0.000002
            self.sim_steps = 10
            self.discount_factor = 1.
        else:
            if mode != 'easy':
                print "Error: CartPole does not recognize mode", mode
                print "Defaulting to 'easy'"
            self.state_range = numpy.array([[-2.4, 2.4],                                 # Cart location bound
                                            [-6., 6.],                                   # Cart velocity bound
                                            [-numpy.pi * 12./180., numpy.pi * 12./180.], # Pole angle bounds
                                            [-6., 6.]])                                  # Pole velocity bound
            self.mu_c = 0.
            self.mu_p = 0.
            self.sim_steps = 1
            self.discount_factor = 0.999

        self.replay_memory = ReplayMemory(history_length=3, memory_size=replay_memory_size,
                                          state_dim=(2 + 2*len(pole_scales), ), state_dtype='float32',
                                          action_dtype='uint8')
        self.reward_range = (-1000., 1.*len(pole_scales)) if self.mode == "swingup" else (-1., 1.)
        self.delta_time = 0.02
        self.max_force = 10.
        self.gravity = -9.8
        self.cart_mass = 1.
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print "Failed to Import Pyplot", e.message
        else:
            self.fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            ax = self.fig.add_subplot(111)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            plt.axis('off')
            plt.axhspan(xmin=0, xmax=1, ymin=0.305, ymax=0.315, facecolor='b', alpha=0.5)
            self.l, = ax.plot([], [], lw=10, color='r')
            self.draw()

    def start(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle.fill(0.0)
        self.pole_velocity.fill(0.0)
        if self.random_start:
            self.pole_angle = (self.npy_rng.rand(*self.pole_angle.shape)-0.5)/5.

    def begin_episode(self, max_episode_step=DEFAULT_MAX_EPISODE_STEP):
        if self.episode_step > self.max_episode_step or self.episode_terminate:
            self.start()
        self.max_episode_step = max_episode_step
        self.episode_reward = 0
        self.episode_step = 0

    def draw(self):
        r = 0.3
        print (self.cart_location + self.state_range[0, 1])/(2*self.state_range[0, 1])
        self.l.set_data([(self.cart_location + self.state_range[0, 1])/(2*self.state_range[0, 1]),
                         (self.cart_location + self.state_range[0, 1])/(2*self.state_range[0, 1])
                  + r*math.cos(self.pole_angle)],
                 [0.31, 0.31 + r*math.sin(self.pole_angle)])
        self.fig.canvas.draw()
        self.fig.show()

    def __gravity_on_pole(self):
        pull = self.mu_p * self.pole_velocity/(self.pole_mass * self.pole_length)
        pull += self.gravity * numpy.sin(self.pole_angle)
        return pull

    def __effective_force(self):
        F = self.pole_mass * self.pole_length * self.pole_velocity**2 * numpy.sin(self.pole_angle)
        F += .75 * self.pole_mass * numpy.cos(self.pole_angle) * self.__gravity_on_pole()
        return F.sum()

    def __effective_mass(self):
        return (self.pole_mass * (1. - .75 * numpy.cos(self.pole_angle)**2)).sum()

    def play(self, a):
        force = self.max_force if a == 1 else -self.max_force
        force += self.max_force*self.npy_rng.normal(scale=self.noise) if self.noise > 0 else 0.0 # Compute noise

        for step in range(self.sim_steps):
            cart_accel = force - self.mu_c * numpy.sign(self.cart_velocity) + self.__effective_force()
            cart_accel /= self.cart_mass + self.__effective_mass()
            pole_accel = (-.75/self.pole_length) * (cart_accel * numpy.cos(self.pole_angle) + self.__gravity_on_pole())

            # Update state variables
            df = (self.delta_time / float(self.sim_steps))
            self.cart_location += df * self.cart_velocity
            self.cart_velocity += df * cart_accel
            self.pole_angle += df * self.pole_velocity
            self.pole_velocity += df * pole_accel

        # If theta (state[2]) has gone past our conceptual limits of [-pi,pi]
        # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
        for i in range(len(self.pole_angle)):
            while self.pole_angle[i] < -numpy.pi:
                self.pole_angle[i] += 2. * numpy.pi

            while self.pole_angle[i] > numpy.pi:
                self.pole_angle[i] -= 2. * numpy.pi

        if self.mode == 'swingup':
            reward = numpy.cos(numpy.abs(self.pole_angle)).sum()
        else:
            reward = -1. if self.episode_terminate else 1.
        self.replay_memory.append(obs=self.get_observation(), action=a, reward=reward,
                                  terminate_flag=self.episode_terminate)
        return reward, self.episode_terminate

    @property
    def episode_terminate(self):
        """Indicates whether or not the episode should terminate.
        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return numpy.abs(self.cart_location) > self.state_range[0,1] \
               or (numpy.abs(self.pole_angle) > self.state_range[2,1]).any() \
               or numpy.abs(self.cart_velocity) > self.state_range[1, 1] \
               or (numpy.abs(self.pole_velocity) > self.state_range[3, 1]).any()

    def get_observation(self):
        return numpy.array([self.cart_location, self.cart_velocity] +
                           self.pole_angle.tolist() + self.pole_velocity.tolist())

