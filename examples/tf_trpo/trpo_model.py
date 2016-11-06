from arena.models.model import ModelWithCritic
import tensorflow as tf
import prettytensor as pt
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg
from baseline import Baseline
from diagonal_gaussian import DiagonalGaussian
import numpy as np
import random
import math
import logging

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


class TrpoModel(ModelWithCritic):
    def __init__(self, observation_space, action_space,
                 min_std=1e-6,
                 subsample_factor=0.8,
                 cg_damping=0.1,
                 cg_iters=20,
                 max_kl=0.01,
                 session=None):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space

        # store constants
        self.min_std = min_std
        self.subsample_factor = subsample_factor
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.max_kl = max_kl

        if session is None:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.session = tf.Session(config=tf.ConfigProto(  # gpu_options=gpu_options,
                log_device_placement=False, ))
        else:
            self.session = session
        self.critic = Baseline(session=self.session, shape=self.ob_space.shape)
        self.distribution = DiagonalGaussian(dim=self.act_space.low.ndim)

        self.theta = None
        self.info_shape = dict(mean=self.act_space.shape,
                               log_std=self.act_space.shape)

        self.net = NetworkContinous(scope="network_continous",
                                    obs_shape=self.ob_space.shape,
                                    action_shape=self.act_space.shape)
        log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n, self.new_dist_info_vars,
                                                              self.old_dist_info_vars)
        surr = -tf.reduce_mean(self.ratio_n * self.net.advant)  # Surrogate loss
        batch_size = tf.shape(self.net.obs)[0]
        batch_size_float = tf.cast(batch_size, tf.float32)
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ent = self.distribution.entropy(self.old_dist_info_vars)
        self.losses = [surr, kl, ent]
        var_list = self.net.var_list
        self.get_flat_params = GetFlat(var_list, session=self.session)  # get theta from var_list
        self.set_params_with_flat_data = SetFromFlat(var_list, session=self.session)  # set theta from var_List
        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars) / batch_size_float
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = list(map(var_shape, var_list))
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        self.gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(tf.reduce_sum(self.gvp), var_list)  # get kl''*p
        self.summary_writer = tf.train.SummaryWriter('./summary', self.session.graph)
        self.session.run(tf.initialize_all_variables())

    def predict(self, observation):
        obs = np.expand_dims(observation, 0)
        action_dist_means_n, action_dist_stds_n = \
            self.session.run([self.net.action_dist_means_n, self.action_dist_stds_n],
                             {self.net.obs: obs})

        rnd = np.random.normal(size=action_dist_means_n[0].shape)
        action = rnd * action_dist_stds_n[0] + action_dist_means_n[0]

        return action, dict(mean=action_dist_means_n[0], log_std=action_dist_stds_n[0])

    def compute_critic(self, states):
        return self.critic.predict(states)

    def compute_update(self, sample_data):
        agent_infos = sample_data["agent_infos"]
        obs_n = sample_data["observations"]
        action_n = sample_data["actions"]
        advant_n = sample_data["advantages"]

        action_dist_means_n = agent_infos["mean"]
        action_dist_logstds_n = agent_infos["log_std"]
        feed = {self.net.obs: obs_n,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n
                }
        self.critic.fit(sample_data)
        thprev = self.get_flat_params()  # get theta_old

        def fisher_vector_product(p):
            feed[self.flat_tangent] = p
            return self.session.run(self.fvp, feed) + self.cg_damping * p

        g = self.session.run(self.pg, feed_dict=feed)
        stepdir = cg(fisher_vector_product, -g, cg_iters=self.cg_iters)
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
        # if shs<0, then the nan error would appear
        lm = np.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)

        def loss(th):
            self.set_params_with_flat_data(th)
            return self.session.run(self.losses, feed_dict=feed)[0]

        theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)

        return None, theta

    def update(self, diff, new=None):
        self.set_params_with_flat_data(new)


# TODO: remove this class
class NetworkContinous(object):
    def __init__(self, scope, obs_shape, action_shape):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                dtype, shape=(None,) + obs_shape, name="%s_obs" % scope)
            self.action_n = tf.placeholder(dtype, shape=(None,) + action_shape, name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)

            self.old_dist_means_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                     name="%s_oldaction_dist_logstds" % scope)
            self.action_dist_means_n = (pt.wrap(self.obs).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc1" % scope).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc2" % scope).
                                        fully_connected(np.prod(action_shape),
                                                        init=tf.random_normal_initializer(-0.05, 0.05),
                                                        name="%s_fc3" % scope))

            # self.N = tf.shape(obs)[0]
            # Nf = tf.cast(self.N, dtype)
            # TODO: STD should be trainable, learn this later
            # TODO: understand this machine code, could be potentially prone to bugs
            self.action_dist_logstd_param = tf.Variable(
                initial_value=(.01 * np.random.randn(1, *action_shape)).astype(np.float32),
                trainable=True, name="%spolicy_logstd" % scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                           {self.obs: obs})
