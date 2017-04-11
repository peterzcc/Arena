from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg
from baseline import Baseline
from diagonal_gaussian import DiagonalGaussian
from network_models import NetworkContinous
import numpy as np
import random
import math
import logging

concat = np.concatenate
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


class MultiTrpoModel(ModelWithCritic):
    def __init__(self, observation_space, action_space,
                 min_std=1e-6,
                 subsample_factor=0.8,
                 cg_damping=0.1,
                 cg_iters=20,
                 max_kl=0.01,
                 timestep_limit=1000,
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

            cpu_config = tf.ConfigProto(
                device_count={'GPU': 0}, log_device_placement=False
            )
            self.session = tf.Session(config=cpu_config)

            # gpu_options = tf.GPUOptions(allow_growth=True)
            # self.session = tf.Session(config=tf.ConfigProto( gpu_options=gpu_options,
            #     log_device_placement=True, ))

        else:
            self.session = session
        self.critic = Baseline(session=self.session, shape=self.ob_space[0].shape,
                               timestep_limit=timestep_limit)
        self.distribution = DiagonalGaussian(dim=self.act_space.low.shape[0])

        self.theta = None
        self.info_shape = dict(mean=self.act_space.shape,
                               log_std=self.act_space.shape,
                               clips=())

        self.net = NetworkContinous(scope="network_continous",
                                    obs_shape=self.ob_space[0].shape,
                                    action_shape=self.act_space.shape)
        # log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.min_std))
        batch_size = tf.shape(self.net.obs)[0]
        self.batch_size_float = tf.cast(batch_size, tf.float32)
        self.action_dist_log_stds_n = self.net.action_dist_logstds_n  # log_std_var
        self.action_dist_std_n = tf.exp(self.action_dist_log_stds_n)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.old_likelihood = self.distribution.log_likelihood_sym(self.net.action_n, self.old_dist_info_vars)

        self.ratio_n = -tf.exp(self.new_likelihood_sym - self.old_likelihood) / self.batch_size_float

        surr = tf.reduce_sum(self.ratio_n * self.net.advant)  # Surrogate loss
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ents = self.distribution.entropy(self.old_dist_info_vars)
        ent = tf.reduce_sum(ents) / self.batch_size_float
        self.losses = [surr, kl, ent]
        var_list = self.net.var_list
        self.get_flat_params = GetFlat(var_list, session=self.session)  # get theta from var_list
        self.set_params_with_flat_data = SetFromFlat(var_list, session=self.session)  # set theta from var_List
        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars) / self.batch_size_float
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
        self.debug = True

    def predict(self, observation):
        if len(observation[0].shape) == len(self.ob_space[0].shape):
            obs = np.expand_dims(observation[0], 0)
        else:
            obs = observation[0]
        action_dist_means_n, action_dist_log_stds_n, action_std_n = \
            self.session.run([self.net.action_dist_means_n, self.action_dist_log_stds_n, self.action_dist_std_n],
                             {self.net.obs: obs})

        rnd = np.random.normal(size=action_dist_means_n[0].shape)
        output = rnd * action_std_n[0] + action_dist_means_n[0]
        action = output  # np.clip(output,self.act_space.low,self.act_space.high).flatten()

        # logging.debug("am:{},\nastd:{}".format(action_dist_means_n[0],action_dist_stds_n[0]))
        if self.debug:
            is_clipped = np.logical_or((action <= self.act_space.low), (action >= self.act_space.high))
            num_clips = np.count_nonzero(is_clipped)
            agent_info = dict(mean=action_dist_means_n[0], log_std=action_dist_log_stds_n[0], clips=num_clips)
        else:
            agent_info = dict(mean=action_dist_means_n[0], log_std=action_dist_log_stds_n[0])
        # logging.debug("tx a:{},\n".format(action))
        return action, agent_info

    def compute_critic(self, states):

        return self.critic.predict(states)

    def compute_update(self, paths):
        # agent_infos = sample_data["agent_infos"]
        # obs_n = sample_data["observations"]
        # action_n = sample_data["actions"]
        # advant_n = sample_data["advantages"]

        # prob_np = concat([path["prob"] for path in paths])  # self._act_prob(ob[None])[0]
        obs_n = concat([np.array([o[0] for o in path["observation"]]) for path in paths])
        action_n = concat([path["action"] for path in paths])
        advant_n = concat([path["advantage"] for path in paths])
        logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))

        action_dist_means_n = concat([path["mean"] for path in paths])
        action_dist_logstds_n = concat([path["log_std"] for path in paths])
        feed = {self.net.obs: obs_n,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n
                }
        self.critic.fit(paths)
        thprev = self.get_flat_params()  # get theta_old

        def fisher_vector_product(p):
            feed[self.flat_tangent] = p
            # print("ratio_n")
            # print(self.session.run(tf.sqrt(tf.reduce_sum(self.ratio_n, feed)**2)),feed)
            return self.session.run(self.fvp, feed) + self.cg_damping * p

        g = self.session.run(
            self.pg,
            feed_dict=feed)
        if self.debug:
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
            logging.debug("act_mean mean: {}".format(np.mean(action_dist_means_n, axis=0)))
            logging.debug("act_mean std: {}".format(np.std(action_dist_means_n, axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))

        stepdir = cg(fisher_vector_product, -g, cg_iters=self.cg_iters)
        sAs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
        # if shs<0, then the nan error would appear
        lm = np.sqrt(sAs / self.max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)
        logging.debug("\nlagrange multiplier:{}\tgnorm:{}\t".format(lm, np.linalg.norm(g)))

        def loss(th):
            self.set_params_with_flat_data(th)
            return self.session.run(self.losses, feed_dict=feed)[0]

        surr_o, kl_o, ent_o = self.session.run(self.losses, feed_dict=feed)
        if self.debug:
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            # logging.debug("\nold theta: {}\n".format(np.linalg.norm(thprev)))
        logging.debug("\nfullstep: {}\n".format(np.linalg.norm(fullstep)))
        theta, d_theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
        self.set_params_with_flat_data(theta)
        surr_new, kl_new, ent_new = self.session.run(self.losses, feed_dict=feed)
        if self.debug:
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            # logging.debug("\nnew theta: {}\nd_theta: {}\n".format(np.linalg.norm(theta), np.linalg.norm(d_theta)))
        return None, None

    def update(self, diff, new=None):
        pass
        # self.set_params_with_flat_data(new)
