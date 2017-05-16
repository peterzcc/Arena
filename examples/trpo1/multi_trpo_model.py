from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched
# from baseline import Baseline
from multi_baseline import MultiBaseline
from diagonal_gaussian import DiagonalGaussian
from network_models import MultiNetwork
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
                 cg_iters=10,
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
        self.minibatch_size = 64
        self.use_empirical_fim = True

        if session is None:

            # cpu_config = tf.ConfigProto(
            #     device_count={'GPU': 0}, log_device_placement=False
            # )
            # self.session = tf.Session(config=cpu_config)

            gpu_options = tf.GPUOptions(allow_growth=True)  # False,per_process_gpu_memory_fraction=0.75)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                            log_device_placement=False))

        else:
            self.session = session
        only_image = True
        n_imgfeat = 4
        self.critic = MultiBaseline(session=self.session, obs_space=self.ob_space,
                                    timestep_limit=timestep_limit, with_image=True, only_image=only_image,
                                    n_imgfeat=n_imgfeat)
        self.distribution = DiagonalGaussian(dim=self.act_space.low.shape[0])

        self.theta = None
        self.info_shape = dict(mean=self.act_space.shape,
                               log_std=self.act_space.shape,
                               clips=())
        self.policy_with_image_input = False
        self.net = MultiNetwork(scope="network_continous",
                                observation_space=self.ob_space,
                                action_shape=self.act_space.shape,
                                with_image=self.policy_with_image_input, only_image=only_image,
                                n_imgfeat=n_imgfeat,
                                extra_feaatures=[self.critic.image_features])
        # log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.min_std))
        batch_size = tf.shape(self.net.state_input)[0]
        self.batch_size_float = tf.cast(batch_size, tf.float32)
        self.action_dist_log_stds_n = self.net.action_dist_logstds_n  # log_std_var
        self.action_dist_std_n = tf.exp(self.action_dist_log_stds_n)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.old_likelihood = self.distribution.log_likelihood_sym(self.net.action_n, self.old_dist_info_vars)

        self.ratio_n = -tf.exp(self.new_likelihood_sym - self.old_likelihood) / self.batch_size_float

        surr = self.surr = tf.reduce_sum(self.ratio_n * self.net.advant)  # Surrogate loss
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ents = self.distribution.entropy(self.old_dist_info_vars)
        ent = tf.reduce_sum(ents) / self.batch_size_float
        self.losses = [surr, kl, ent]
        if self.policy_with_image_input:
            self.infos = [tf.reduce_mean(self.net.image_features)]
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

        # splitted_l = self.new_likelihood_sym.unpack()
        # splitted_l = [tf.slice(self.new_likelihood_sym, begin=i, size=1) for i in range(self.minibatch_size)]
        # def grad_varlist(x):
        #     return flatgrad(x, var_list)
        # grad_l_per_sample = tf.map_fn(grad_varlist,splitted_l)
        # def batch_fvp(g):
        #     return g * tf.reduce_sum(g*self.flat_tangent)
        # batch_g_gT_x = tf.map_fn(batch_fvp, grad_l_per_sample)
        # self.agg_g_gT_x = tf.reduce_sum(batch_g_gT_x, 0)/self.batch_size_float

        self.is_real_data = tf.placeholder(shape=(self.minibatch_size,), dtype=tf.float32)



        grad_loglikelihood = flatgrad(self.new_likelihood_sym, var_list)
        list_logl = tf.unstack(self.new_likelihood_sym, num=self.minibatch_size)
        batch_grad = tf.stack([flatgrad(l, var_list) for l in list_logl]) * tf.expand_dims(self.is_real_data, axis=1)
        batch_gT_x = tf.matmul(batch_grad, tf.expand_dims(self.flat_tangent, axis=1))
        self.batch_g_gT_x = tf.reshape(tf.matmul(batch_grad, batch_gT_x, transpose_a=True, transpose_b=False), [-1])


        gT_x = tf.reduce_sum(grad_loglikelihood * self.flat_tangent)
        self.g_gT_x = grad_loglikelihood * gT_x

        # TODO: implement empirical
        self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)
        self.session.run(tf.global_variables_initializer())
        self.update_critic = True
        self.debug = True

    def run_batched_fvp(self, func_batch, func_single, feed, N, session, minibatch_size=64, extra_input={}):
        result = None
        for start in range(0, N, minibatch_size):  # TODO: verify this
            end = min(start + minibatch_size, N)
            this_size = end - start

            if this_size == minibatch_size:
                slc = range(start, end)
                this_feed = {k: v[slc] for k, v in list(feed.items())}
                this_result = \
                    np.array(session.run(func_batch, feed_dict={**this_feed, **extra_input,
                                                                self.is_real_data: np.ones(minibatch_size)}))
            else:
                slc = range(start, end)
                is_real_data = np.zeros(minibatch_size)
                is_real_data[0:this_size] = 1
                this_feed = \
                    {k:
                         np.concatenate([v[slc], np.zeros(shape=(minibatch_size - this_size,) + v.shape[1:])])
                     for k, v in list(feed.items())}
                this_result = \
                    np.array(session.run(func_single, feed_dict={**this_feed, **extra_input,
                                                                 self.is_real_data: is_real_data}))
            if result is None:
                result = this_result
            else:
                result += this_result
        result /= N
        return result

    def predict(self, observation):
        if len(observation[0].shape) == len(self.ob_space[0].shape):
            obs = [np.expand_dims(observation[0], 0), np.expand_dims(observation[1], 0)]
        else:
            obs = observation
        action_dist_means_n, action_dist_log_stds_n, action_std_n = \
            self.session.run([self.net.action_dist_means_n, self.action_dist_log_stds_n, self.action_dist_std_n],
                             {self.net.state_input: obs[0], self.net.img_input: obs[1], self.critic.img_input: obs[1]})

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

        if self.update_critic:
            self.critic.fit(paths)
            self.update_critic = False
            return None, None
        else:
            self.update_critic = True

        state_input = concat([path["observation"][0] for path in paths])
        img_input = concat([path["observation"][1] for path in paths])
        action_n = concat([path["action"] for path in paths])
        advant_n = concat([path["advantage"] for path in paths])
        logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))

        action_dist_means_n = concat([path["mean"] for path in paths])
        action_dist_logstds_n = concat([path["log_std"] for path in paths])
        feed = {self.net.state_input: state_input,
                self.net.img_input: img_input,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n,
                self.critic.img_input: img_input
                }
        batch_size = advant_n.shape[0]
        thprev = self.get_flat_params()  # get theta_old
        if self.use_empirical_fim:
            def fisher_vector_product(p):
                # feed[self.flat_tangent] = p

                fvp = self.run_batched_fvp(self.batch_g_gT_x, self.g_gT_x, feed, batch_size, self.session,
                                           minibatch_size=self.minibatch_size,
                                           extra_input={self.flat_tangent: p})
                return fvp + self.cg_damping * p
        else:
            def fisher_vector_product(p):
                # feed[self.flat_tangent] = p

                fvp = run_batched(self.fvp, feed, batch_size, self.session, minibatch_size=self.minibatch_size,
                                  extra_input={self.flat_tangent: p})
                return fvp + self.cg_damping * p

        g = run_batched(self.pg, feed, batch_size, self.session, minibatch_size=self.minibatch_size)
        if self.debug:
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
            logging.debug("act_mean mean: {}".format(np.mean(action_dist_means_n, axis=0)))
            logging.debug("act_mean std: {}".format(np.std(action_dist_means_n, axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
            if self.policy_with_image_input:
                img_features = run_batched(self.infos, feed, batch_size, self.session,
                                           minibatch_size=self.minibatch_size)
                logging.debug("infos: {}".format(img_features))

        stepdir = cg(fisher_vector_product, -g, cg_iters=self.cg_iters)
        sAs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
        # if shs<0, then the nan error would appear
        lm = np.sqrt(sAs / self.max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)
        logging.debug("\nlagrange multiplier:{}\tgnorm:{}\t".format(lm, np.linalg.norm(g)))

        def loss(th):
            self.set_params_with_flat_data(th)
            surr = run_batched(self.surr, feed, batch_size, self.session, minibatch_size=self.minibatch_size,
                               extra_input={})
            return surr

        if self.debug:
            surr_o, kl_o, ent_o = run_batched(self.losses, feed, batch_size, self.session,
                                              minibatch_size=self.minibatch_size)
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            # logging.debug("\nold theta: {}\n".format(np.linalg.norm(thprev)))
        logging.debug("\nfullstep: {}\n".format(np.linalg.norm(fullstep)))
        theta, d_theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
        self.set_params_with_flat_data(theta)

        if self.debug:
            surr_new, kl_new, ent_new = run_batched(self.losses, feed, batch_size, self.session,
                                                    minibatch_size=self.minibatch_size)
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            # logging.debug("\nnew theta: {}\nd_theta: {}\n".format(np.linalg.norm(theta), np.linalg.norm(d_theta)))
        return None, None

    def update(self, diff, new=None):
        pass
        # self.set_params_with_flat_data(new)
