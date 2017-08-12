from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, stochastic_cg
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
                 cg_damping=0.1,
                 cg_iters=10,
                 max_kl=0.01,
                 timestep_limit=1000,
                 session=None):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space
        logging.debug("\naction space: {} to {}".format(action_space.low, action_space.high))
        logging.debug("\nstate space: {} to {}".format(observation_space[0].low, observation_space[0].high))

        # store constants
        self.min_std = min_std
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.max_kl = max_kl
        self.minibatch_size = 256
        self.use_empirical_fim = True
        self.real_start = 1e7

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
        n_imgfeat = 32
        self.critic_append_image = False
        self.policy_with_image_input = False
        self.critic = MultiBaseline(session=self.session, obs_space=self.ob_space,
                                    timestep_limit=timestep_limit, with_image=self.critic_append_image,
                                    n_imgfeat=n_imgfeat, hidden_sizes=(64, 64),
                                    conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)))
        # conv_sizes=(((3, 3), 32, 1), ((3, 3), 64, 1)))
        self.distribution = DiagonalGaussian(dim=self.act_space.low.shape[0])

        self.theta = None
        self.info_shape = dict(mean=self.act_space.shape,
                               log_std=self.act_space.shape,
                               clips=(),
                               img_enabled=(),
                               st_enabled=self.ob_space[0].low.shape)

        self.net = MultiNetwork(scope="state_agent",
                                observation_space=self.ob_space,
                                action_shape=self.act_space.shape,
                                with_image=self.policy_with_image_input,
                                n_imgfeat=n_imgfeat,
                                extra_feaatures=[np.zeros((4,), dtype=np.float32)],  # [],  #
                                conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)),  #(((3, 3), 2, 2),),  #
                                )
        self.real_net = MultiNetwork(scope="img_agent",
                                     observation_space=self.ob_space,
                                     action_shape=self.act_space.shape,
                                     with_image=self.policy_with_image_input,
                                     n_imgfeat=4,
                                     extra_feaatures=[],  #*self.critic.pre_image_features],#
                                     conv_sizes=(((4, 4), 64, 1), ((3, 3), 64, 1)),  #
                                     )
        self.imi_loss = tf.reduce_mean(tf.square(self.net.action_dist_means_n - self.real_net.action_dist_means_n))
        self.imi_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.imi_train = self.imi_opt.minimize(self.imi_loss,
                                               aggregation_method=tf.AggregationMethod.DEFAULT,
                                               var_list=self.real_net.var_list)
        log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.min_std))
        batch_size = tf.shape(self.net.state_input)[0]
        self.batch_size_float = tf.cast(batch_size, tf.float32)
        self.action_dist_log_stds_n = log_std_var  #self.net.action_dist_logstds_n  #
        self.action_dist_std_n = tf.exp(self.action_dist_log_stds_n)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.old_likelihood = self.distribution.log_likelihood_sym(self.net.action_n, self.old_dist_info_vars)

        self.ratio_n = tf.exp(self.new_likelihood_sym - self.old_likelihood)
        self.p_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.net.var_list])
        self.k_p_l2 = 0.01
        self.img_feature_norm = tf.reduce_mean(tf.square(self.real_net.image_features))
        self.PPO_eps = 0.2
        self.clipped_ratio = tf.clip_by_value(self.ratio_n,1.0-self.PPO_eps,1.0+self.PPO_eps)
        raw_surr = self.ratio_n * self.net.advant
        clipped_surr = self.clipped_ratio * self.net.advant
        surr = self.surr = -tf.reduce_mean(tf.minimum(raw_surr,
                                                      clipped_surr))  # Surrogate loss

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
        self.is_real_data = tf.placeholder(shape=(self.minibatch_size,), dtype=tf.float32)

        grad_loglikelihood = flatgrad(self.new_likelihood_sym, var_list)
        list_logl = tf.unstack(self.new_likelihood_sym, num=self.minibatch_size)
        batch_grad = tf.stack([flatgrad(l, var_list) for l in list_logl]) * tf.expand_dims(self.is_real_data, axis=1)
        batch_gT_x = tf.matmul(batch_grad, tf.expand_dims(self.flat_tangent, axis=1))
        self.batch_g_gT_x = tf.reshape(tf.matmul(batch_grad, batch_gT_x, transpose_a=True, transpose_b=False), [-1])

        self.vanila_surr = - tf.reduce_mean(self.new_likelihood_sym * self.net.advant)
        self.vanila_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="step_size")
        self.vanila_opt = tf.train.AdamOptimizer(learning_rate=self.vanila_sym_step_size,
                                                 beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.vanila_train = self.vanila_opt.minimize(self.vanila_surr, var_list=var_list)
        self.vanila_step_size = 1e-3
        self.vanila_losses = [self.vanila_surr, kl, ent]
        self.vn_grad_list = tf.gradients(self.vanila_surr, var_list)
        self.vn_grad_placeholders = [tf.placeholder(tf.float32, shape=g.shape, name="dummy")
                                     for g in self.vn_grad_list]
        self.vn_apply_grad = self.vanila_opt.apply_gradients(grads_and_vars=zip(self.vn_grad_placeholders, var_list))
        self.v_ds_kl = 2e-3
        self.use_trpo = True


        gT_x = tf.reduce_sum(grad_loglikelihood * self.flat_tangent)
        self.g_gT_x = grad_loglikelihood * gT_x
        self.saved_paths = []

        # self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)
        self.session.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver([*self.net.var_list,*self.critic.var_list])
        self.saver = tf.train.Saver(var_list=self.net.var_list)
        self.init_model_path = self.saver.save(self.session, 'init_model')
        self.n_update = 0
        self.separate_update = True
        self.update_critic = True
        self.update_policy = True
        self.debug = True

    def fvp_minibatch(self, func_batch, feed, slc, session, minibatch_size=64, extra_input={}):
        this_feed = {k: v[slc] for k, v in list(feed.items())}
        this_size = len(slc)
        if this_size == minibatch_size:
            this_feed = {k: v[slc] for k, v in list(feed.items())}
            this_result = \
                np.array(session.run(func_batch, feed_dict={**this_feed, **extra_input,
                                                            self.is_real_data: np.ones(minibatch_size)}))
        else:
            is_real_data = np.zeros(minibatch_size)
            is_real_data[0:this_size] = 1
            this_feed = \
                {k:
                     np.concatenate([v[slc], np.zeros(shape=(minibatch_size - this_size,) + v.shape[1:])])
                 for k, v in list(feed.items())}
            this_result = \
                np.array(session.run(func_batch, feed_dict={**this_feed, **extra_input,
                                                            self.is_real_data: is_real_data}))
        return this_result

    def run_batched_fvp(self, func_batch, feed, N, session, minibatch_size=64, extra_input={}):
        result = None
        for start in range(0, N, minibatch_size):
            end = min(start + minibatch_size, N)
            # this_size = end - start
            slc = range(start, end)
            this_result = self.fvp_minibatch(func_batch, feed, slc, session, minibatch_size, extra_input)
            if result is None:
                result = this_result
            else:
                result += this_result
        result /= N
        return result

    def run_lissa(self, func_batch, v, feed, N, session, minibatch_size=64, lr=1.0,
                  recursion_depth=80,
                  initial_guess=None,
                  verbose=True):
        num_samples = 1
        log_iter = 10
        inverse_hvp = None
        for i in range(num_samples):
            fmtstr = "%10i %10.3g %10.3g"
            titlestr = "%10s %10s %10s"
            if verbose: logging.debug(titlestr % ("iter", "residual norm", "soln norm"))
            cur_estimate = v if initial_guess is None else initial_guess
            j = 0
            while j < recursion_depth:
                training_inds = np.concatenate([np.random.permutation(N), np.random.permutation(N)])
                for start in range(0, N, self.minibatch_size):
                    end = start + self.minibatch_size
                    this_size = start - end
                    slc = training_inds[range(start, end)]
                    h_inv_x = self.fvp_minibatch(func_batch, feed, slc, session, minibatch_size,
                                                 extra_input={self.flat_tangent: cur_estimate})
                    diff = v - self.cg_damping * cur_estimate - h_inv_x * 1.0 / this_size
                    # porp = np.mean(np.linalg.norm(v) /np.linalg.norm(h_inv_x))

                    if verbose and ((j % log_iter == 0) or (j == recursion_depth - 1)):
                        # r = v - self.cg_damping*cur_estimate -\
                        #     self.run_batched_fvp(func_batch, feed, N, session, minibatch_size,
                        #                                 extra_input={self.flat_tangent: cur_estimate})
                        logging.debug(fmtstr % (j, diff.dot(diff), np.linalg.norm(cur_estimate)))
                    cur_estimate = cur_estimate + lr * diff

                    j += 1
                    if j >= recursion_depth:
                        break
            if inverse_hvp is None:
                inverse_hvp = cur_estimate * 1.0 / scale
            else:
                inverse_hvp += cur_estimate * 1.0 / scale
        inverse_hvp /= num_samples
        return inverse_hvp


    def get_state_activation(self, t_batch):
        # start_ratio = 0.1
        # end_ratio = 0.0
        # start_n_batch = 100
        # final_n_batch = 350
        # noise_k = 0.0 if t_batch < start_n_batch else \
        #     ((t_batch-start_n_batch)/(final_n_batch-start_n_batch) if t_batch < final_n_batch else 1.0)
        # ratio = noise_k * end_ratio + (1 - noise_k)*start_ratio
        # is_enabled = (np.random.random_sample(size=None) < ratio)

        is_enabled = True  # (t_batch < 20)

        st_enabled = np.array([1.0, 1.0, 1.0, 1.0]) if is_enabled else np.array([0.0, 0.0, 0.0, 0.0])
        img_enabled = 1.0 - is_enabled
        # st_enabled = np.array([0.0, 0.0])
        return st_enabled, img_enabled

    def predict(self, observation):
        if len(observation[0].shape) == len(self.ob_space[0].shape):
            obs = [np.expand_dims(observation[0], 0), np.expand_dims(observation[1], 0)]
        else:
            obs = observation
        st_enabled, img_enabled = self.get_state_activation(self.n_update)

        exp_st_enabled = np.expand_dims(st_enabled, 0)
        exp_img_enabled = np.expand_dims(img_enabled, 0)

        if self.n_update > self.real_start:
            action_dist_means_n, action_dist_log_stds_n, action_std_n = \
                self.session.run(
                    [self.real_net.action_dist_means_n, self.action_dist_log_stds_n, self.action_dist_std_n],
                    {self.real_net.state_input: obs[0], self.real_net.img_input: obs[1],
                     self.net.state_input: obs[0], self.net.img_input: obs[1],
                     self.net.st_enabled: exp_st_enabled,
                     self.net.img_enabled: exp_img_enabled,
                     self.real_net.st_enabled: np.zeros_like(exp_st_enabled),
                     self.real_net.img_enabled: np.ones_like(exp_img_enabled)
                     })
        else:
            action_dist_means_n, action_dist_log_stds_n, action_std_n = \
                self.session.run([self.net.action_dist_means_n, self.action_dist_log_stds_n, self.action_dist_std_n],
                                 {self.net.state_input: obs[0], self.net.img_input: obs[1],
                                  self.net.st_enabled: exp_st_enabled,
                                  self.net.img_enabled: exp_img_enabled,
                                  })


        rnd = np.random.normal(size=action_dist_means_n[0].shape)
        output = rnd * action_std_n[0] + action_dist_means_n[0]
        action = output  #np.clip(output, self.act_space.low, self.act_space.high).flatten()

        # logging.debug("am:{},\nastd:{}".format(action_dist_means_n[0],action_dist_stds_n[0]))
        agent_info = dict(mean=action_dist_means_n[0], log_std=action_dist_log_stds_n[0],
                          st_enabled=st_enabled, img_enabled=img_enabled)
        if self.debug:
            is_clipped = np.logical_or((action <= self.act_space.low), (action >= self.act_space.high))
            num_clips = np.count_nonzero(is_clipped)
            agent_info["clips"] = num_clips

        # logging.debug("tx a:{},\n".format(action))

        return action, agent_info

    def concat_paths(self, paths, real_feed=False):
        state_input = concat([path["observation"][0] for path in paths])
        img_input = concat([path["observation"][1] for path in paths])
        times = concat([path["times"] for path in paths], axis=0)
        returns = concat([path["return"] for path in paths])
        img_enabled = concat([path["img_enabled"] for path in paths])
        st_enabled = concat([path["st_enabled"] for path in paths])
        action_n = concat([path["action"] for path in paths])
        advant_n = concat([path["advantage"] for path in paths])
        action_dist_means_n = concat([path["mean"] for path in paths])
        action_dist_logstds_n = concat([path["log_std"] for path in paths])
        feed = {self.net.state_input: state_input,
                self.net.img_input: img_input,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n,
                # self.critic.img_input: img_input,
                # self.critic.st_enabled: st_enabled,
                # self.critic.img_enabled: img_enabled,
                self.net.st_enabled: st_enabled,
                self.net.img_enabled: img_enabled,
                # self.real_net.state_input: state_input,
                # self.real_net.img_input: img_input,
                # self.real_net.st_enabled: np.zeros_like(st_enabled),
                # self.real_net.img_enabled: np.ones_like(img_enabled),
                }
        # if not real_feed:
        #     feed = {self.net.state_input: state_input,
        #             self.net.img_input: img_input,
        #             self.net.advant: advant_n,
        #             self.net.old_dist_means_n: action_dist_means_n,
        #             self.net.old_dist_logstds_n: action_dist_logstds_n,
        #             self.net.action_n: action_n,
        #             # self.critic.img_input: img_input,
        #             # self.critic.st_enabled: st_enabled,
        #             # self.critic.img_enabled: img_enabled,
        #             self.net.st_enabled: st_enabled,
        #             self.net.img_enabled: img_enabled,
        #             self.real_net.state_input: state_input,
        #             self.real_net.img_input: img_input,
        #             self.real_net.st_enabled: np.zeros_like(st_enabled),
        #             self.real_net.img_enabled: np.ones_like(img_enabled),
        #             }
        # else:
        #     feed = {self.net.state_input: state_input,
        #             self.net.img_input: img_input,
        #             # self.net.advant: advant_n,
        #             # self.net.old_dist_means_n: action_dist_means_n,
        #             # self.net.old_dist_logstds_n: action_dist_logstds_n,
        #             # self.net.action_n: action_n,
        #             # self.critic.img_input: img_input,
        #             # self.critic.st_enabled: st_enabled,
        #             # self.critic.img_enabled: img_enabled,
        #             self.net.st_enabled: st_enabled,
        #             self.net.img_enabled: img_enabled,
        #             self.real_net.state_input: state_input,
        #             self.real_net.img_input: img_input,
        #             self.real_net.st_enabled: np.zeros_like(st_enabled),
        #             self.real_net.img_enabled: np.ones_like(img_enabled),
        #             }
        path_dict = {"state_input": state_input,
                     "img_input": img_input,
                     "times": times,
                     "returns": returns,
                     "img_enabled": img_enabled,
                     "st_enabled": st_enabled}
        return feed, path_dict

    def compute_critic(self, states):
        return self.critic.predict(states)

    def compute_update(self, paths):
        # if self.separate_update:
        #     if self.n_update % 4< 2 :
        #         self.update_critic = True
        #         self.update_policy = False
        #     else:
        #         self.update_critic = False
        #         self.update_policy = True
        if self.n_update == self.real_start + 1:
            self.saver.restore(self.session, "pretrain_model")
            self.pretrain_model_path = self.saver.save(self.session, 'pretrain_model')
            logging.info("model_path: {}".format(self.pretrain_model_path))
        self.update_critic = not self.n_update > self.real_start
        self.update_policy = True

        # if self.n_update % 2 != 0 or self.n_update < 20:
        #     self.update_critic = False
        #     self.update_policy = False
        #     self.critic.fit(path_dict, update_mode="img", num_pass=2)
        # else:
        #     self.update_critic = True
        #     self.update_policy = True
        feed, path_dict = self.concat_paths(paths, real_feed=(self.n_update > self.real_start))

        # if self.n_update < 20:
        #     self.saved_paths = [*self.saved_paths, *paths]
        # elif self.n_update == 20:
        #     # self.saver.restore(self.session, 'pretrained_model')
        #
        #     self.saver.restore(self.session,self.init_model_path)
        #     _, img_path_dict = self.concat_paths(self.saved_paths)
        #     self.saved_paths = []
        #     self.critic.fit(img_path_dict, update_mode="img", num_pass=5)
        #     self.saver.save(self.session, 'pretrained_model')
        # else:
        #     self.update_critic = True
        #     self.update_policy = True


        if self.update_critic:
            self.critic.fit(path_dict, update_mode="full", num_pass=2)
        self.n_update += 1

        # logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))
        advant_n = feed[self.net.advant]
        state_input = feed[self.net.state_input]
        action_dist_logstds_n = feed[self.net.old_dist_logstds_n]
        batch_size = advant_n.shape[0]
        thprev = self.get_flat_params()  # get theta_old
        if self.debug:
            logging.debug("state max: {}\n min: {}".format(state_input.max(axis=0), state_input.min(axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
        if not self.update_policy:
            return None, None

        if self.n_update > self.real_start:
            if self.debug:
                imi_loss = run_batched(self.imi_loss, feed, batch_size, self.session,
                                       minibatch_size=self.minibatch_size)
                logging.debug("\nimi_loss: {}".format(imi_loss))

            training_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                if start > batch_size - 2 * self.minibatch_size:
                    end = batch_size
                else:
                    end = start + self.minibatch_size
                slc = training_inds[range(start, end)]
                this_feed = {k: v[slc] for k, v in list(feed.items())}
                self.session.run(self.imi_train, feed_dict=this_feed)
                if end == batch_size:
                    break

            if self.debug:
                imi_loss, img_norm = run_batched([self.imi_loss, self.img_feature_norm], feed, batch_size, self.session,
                                                 minibatch_size=self.minibatch_size)
                logging.debug("\nimi_loss: {}\n img_norm: {}".format(imi_loss, img_norm))
            return None, None

        if not self.use_trpo:
            surr_o, kl_o, ent_o = run_batched(self.vanila_losses, feed, batch_size, self.session,
                                              minibatch_size=self.minibatch_size)
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))

            # _ = self.session.run(self.vanila_train, feed_dict={**feed,
            #                                                    self.vanila_sym_step_size:self.vanila_step_size})
            grads = run_batched(self.vn_grad_list, feed, batch_size, self.session,
                                minibatch_size=self.minibatch_size)
            grad_dict = {p: v for (p, v) in zip(self.vn_grad_placeholders, grads)}
            _ = self.session.run(self.vn_apply_grad,
                                 feed_dict={**grad_dict, self.vanila_sym_step_size: self.vanila_step_size})
            surr_new, kl_new, ent_new = run_batched(self.vanila_losses, feed, batch_size, self.session,
                                                    minibatch_size=self.minibatch_size)
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            if kl_new > self.v_ds_kl * 2:
                self.vanila_step_size /= 1.5
                print('stepsize -> %s' % self.vanila_step_size)
            elif kl_new < self.v_ds_kl / 2:
                self.vanila_step_size *= 1.5
                print('stepsize -> %s' % self.vanila_step_size)
            else:
                print('stepsize OK')

            return None, None

        if self.use_empirical_fim:
            def fisher_vector_product(p):
                # feed[self.flat_tangent] = p

                fvp = self.run_batched_fvp(self.batch_g_gT_x, feed, batch_size, self.session,
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

        stepdir = cg(fisher_vector_product, -g, cg_iters=self.cg_iters, residual_tol=1e-5)
        test_stepdir = self.run_lissa(self.batch_g_gT_x, -g, feed, batch_size, self.session, self.minibatch_size,
                                      recursion_depth=5000, initial_guess=stepdir, lr=0.0)
        # test_stepdir = stochastic_cg(fisher_vector_product, -g, cg_iters=100, residual_tol=1e-10,
        #                              initial_guess=stepdir,scale=25.0)
        sAs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
        # if shs<0, then the nan error would appear
        lm = np.sqrt(sAs / self.max_kl)
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)
        logging.debug("\nlagrange multiplier:{}\tgnorm:{}\t".format(lm, np.linalg.norm(g)))

        def loss(th):
            self.set_params_with_flat_data(th)
            surr = run_batched(self.surr, feed, batch_size, self.session, minibatch_size=self.minibatch_size)
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
