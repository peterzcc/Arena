from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st
# from baseline import Baseline
from multi_baseline import MultiBaseline
from diagonal_gaussian import DiagonalGaussian
from network_models import MultiNetwork
import numpy as np
import random
import threading as thd
import logging
from dict_memory import DictMemory
from read_write_lock import ReadWriteLock
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
                 n_imgfeat=None,
                 target_kl=0.003,
                 minibatch_size=128,
                 gamma=0.995,
                 gae_lam=0.97,
                 mode="ADA_KL",
                 num_actors=1,
                 batch_size=20,
                 batch_mode="episode"):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space
        logging.debug("\naction space: {} to {}".format(action_space.low, action_space.high))
        logging.debug("\nstate space: {} to {}".format(observation_space[0].low, observation_space[0].high))
        args = locals()
        logging.debug("model args:\n {}".format(args))

        # store constants
        self.min_std = min_std
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.max_kl = max_kl
        self.minibatch_size = minibatch_size
        self.use_empirical_fim = True
        self.mode = mode
        self.policy_lock = ReadWriteLock()
        self.critic_lock = ReadWriteLock()
        self.num_actors = num_actors
        self.execution_barrier = thd.Barrier(num_actors)
        self.act_means = None
        self.act_logstds = None
        self.obs_list = [None for i in range(num_actors)]
        self.act_list = [None for i in range(num_actors)]
        self.batch_mode = batch_mode  # "episode" #"timestep"
        self.batch_size = batch_size

        # cpu_config = tf.ConfigProto(
        #     device_count={'GPU': 0}, log_device_placement=False
        # )
        # self.session = tf.Session(config=cpu_config)
        self.memory = DictMemory(gamma=gamma, lam=gae_lam, use_gae=True, normalize=True,
                                 timestep_limit=timestep_limit,
                                 f_critic=self.compute_critic,
                                 num_actors=num_actors,
                                 f_check_batch=self.check_batch_finished)

        gpu_options = tf.GPUOptions(allow_growth=True)  # False,per_process_gpu_memory_fraction=0.75)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                            log_device_placement=False))

        self.n_imgfeat = n_imgfeat if n_imgfeat is not None else self.ob_space[0].shape[0]
        self.comb_method = aggregate_feature

        hid1_size = observation_space[0].shape[0] * 10
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        hidden_sizes = (hid1_size, hid2_size, hid3_size)
        self.critic = MultiBaseline(session=self.session, obs_space=self.ob_space,
                                    timestep_limit=timestep_limit,
                                    activation=tf.tanh,
                                    n_imgfeat=self.n_imgfeat, hidden_sizes=hidden_sizes,
                                    conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)),
                                    comb_method=self.comb_method)

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
                                n_imgfeat=self.n_imgfeat,
                                extra_feaatures=[],
                                # [],  #[np.zeros((4,), dtype=np.float32)],  #
                                conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)),  #(((3, 3), 2, 2),),  #
                                comb_method=self.comb_method
                                )

        log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(self.min_std))
        batch_size = tf.shape(self.net.state_input)[0]
        self.batch_size_float = tf.cast(batch_size, tf.float32)
        self.action_dist_log_stds_n = log_std_var  #self.net.action_dist_logstds_n  #
        # self.action_dist_std_n = tf.exp(self.action_dist_log_stds_n)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.old_likelihood = self.distribution.log_likelihood_sym(self.net.action_n, self.old_dist_info_vars)

        self.ratio_n = tf.exp(self.new_likelihood_sym - self.old_likelihood)
        self.p_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.net.var_list])
        self.k_p_l2 = 0.01
        self.PPO_eps = 0.2
        self.clipped_ratio = tf.clip_by_value(self.ratio_n,1.0-self.PPO_eps,1.0+self.PPO_eps)
        raw_surr = self.ratio_n * self.net.advant
        clipped_surr = self.clipped_ratio * self.net.advant
        surr = self.surr = -tf.reduce_mean(tf.minimum(raw_surr,
                                                      clipped_surr))  # Surrogate loss

        kl = self.kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
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
        self.is_real_data = tf.placeholder(shape=(self.minibatch_size,), dtype=tf.float32)

        grad_loglikelihood = flatgrad(self.new_likelihood_sym, var_list)
        list_logl = tf.unstack(self.new_likelihood_sym, num=self.minibatch_size)
        if self.mode == "TRPO":
            batch_grad = tf.stack([flatgrad(l, var_list) for l in list_logl]) * tf.expand_dims(self.is_real_data,
                                                                                               axis=1)
            batch_gT_x = tf.matmul(batch_grad, tf.expand_dims(self.flat_tangent, axis=1))
            self.batch_g_gT_x = tf.reshape(tf.matmul(batch_grad, batch_gT_x, transpose_a=True, transpose_b=False), [-1])
        self.target_kl = target_kl


        # adaptive kl
        self.a_beta = tf.placeholder(shape=[], dtype=tf.float32, name="a_beta")
        self.a_beta_value = 3
        self.a_eta_value = 50
        self.a_surr = -tf.reduce_mean(raw_surr) + self.a_beta * kl + \
                      self.a_eta_value * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.target_kl))
        self.a_step_size = 0.0001
        self.a_max_step_size = 0.01
        self.a_min_step_size = 1e-7
        self.a_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="a_step_size")
        self.a_opt = tf.train.AdamOptimizer(learning_rate=self.a_sym_step_size)
        # self.a_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # self.a_train = self.a_opt.minimize(self.a_surr,var_list=var_list)
        self.a_grad_list = tf.gradients(self.a_surr, var_list)
        self.a_grad_placeholders = [tf.placeholder(tf.float32, shape=g.shape, name="a_grad_sym")
                                    for g in self.a_grad_list]
        self.a_old_parameters = [tf.Variable(tf.zeros(v.shape, dtype=tf.float32), name="a_old_param") for v in var_list]
        self.a_backup_op = [tf.assign(old_v, v) for (old_v, v) in zip(self.a_old_parameters, var_list)]
        self.a_rollback_op = [[tf.assign(v, old_v) for (old_v, v) in zip(self.a_old_parameters, var_list)]]
        self.a_apply_grad = self.a_opt.apply_gradients(grads_and_vars=zip(self.a_grad_placeholders, var_list))
        self.a_losses = [self.a_surr, kl, ent]
        self.a_beta_max = 35.0
        self.a_beta_min = 1.0 / 35.0
        self.update_per_epoch = 20

        gT_x = tf.reduce_sum(grad_loglikelihood * self.flat_tangent)
        self.g_gT_x = grad_loglikelihood * gT_x
        self.saved_paths = []

        # self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if len(self.critic.img_var_list) > 0:
            self.feat_saver = tf.train.Saver(var_list=self.critic.img_var_list)
        # self.saver = tf.train.Saver(var_list=self.net.var_list)
        self.init_model_path = self.saver.save(self.session, 'init_model')
        self.n_update = 0
        self.n_pretrain = 0
        self.separate_update = True
        self.update_critic = True
        self.update_policy = True
        self.debug = True

    def get_state_activation(self, t_batch):
        # start_ratio = 1.0
        # end_ratio = 0.0
        # start_n_batch = 90
        # final_n_batch = 140
        # noise_k = 0.0 if t_batch < start_n_batch else \
        #     ((t_batch-start_n_batch)/(final_n_batch-start_n_batch) if t_batch < final_n_batch else 1.0)
        # ratio = noise_k * end_ratio + (1 - noise_k)*start_ratio
        # is_enabled = (np.random.random_sample(size=None) < ratio)

        # is_enabled = t_batch < self.n_pretrain
        #
        # is_enabled = False
        # st_enabled = np.array([1.0, 1.0, 1.0, 1.0]) if is_enabled else np.array([0.0, 0.0, 0.0, 0.0])
        # img_enabled = 1.0 - is_enabled


        all_st_enabled = True
        st_enabled = np.ones(self.ob_space[0].shape) if all_st_enabled else np.zeros(self.ob_space[0].shape)
        img_enabled = np.array((1.0 - all_st_enabled,))
        return st_enabled, img_enabled

    def predict(self, observation, pid=0):

        if self.num_actors == 1:
            if len(observation[0].shape) == len(self.ob_space[0].shape):
                obs = [np.expand_dims(observation[0], 0)]
                if self.n_imgfeat > 0:
                    obs.append(np.expand_dims(observation[1], 0))
            else:
                obs = observation
            st_enabled, img_enabled = self.get_state_activation(self.n_update)
            exp_st_enabled = np.expand_dims(st_enabled, 0)
            exp_img_enabled = img_enabled
        else:
            self.obs_list[pid] = observation
            self.execution_barrier.wait()
            obs = [np.stack([ob[0] for ob in self.obs_list])]
            if self.n_imgfeat > 0:
                obs.append(np.stack([o[1] for o in self.obs_list]))
            st_enabled, img_enabled = self.get_state_activation(self.n_update)
            st_enabled = np.tile(st_enabled, (self.num_actors, 1))
            img_enabled = np.repeat(img_enabled, self.num_actors, axis=0)
            exp_st_enabled = st_enabled
            exp_img_enabled = img_enabled



        feed = {self.net.state_input: obs[0],
                self.critic.st_enabled: exp_st_enabled,
                self.critic.img_enabled: exp_img_enabled,
                self.net.st_enabled: exp_st_enabled,
                self.net.img_enabled: exp_img_enabled,
                }
        if self.n_imgfeat > 0:
            feed[self.critic.img_input] = obs[1]
            feed[self.net.img_input] = obs[1]
        if pid == 0:
            self.policy_lock.acquire_read()
            action_dist_means_n, action_dist_log_stds_n = \
                self.session.run([self.net.action_dist_means_n, self.action_dist_log_stds_n],
                                 feed)
            self.act_means = action_dist_means_n
            self.act_logstds = action_dist_log_stds_n
            self.policy_lock.release_read()
        self.execution_barrier.wait()

        # logging.debug("am:{},\nastd:{}".format(action_dist_means_n[0],action_dist_stds_n[0]))

        agent_info = dict(mean=self.act_means[pid, :], log_std=self.act_logstds,
                          st_enabled=exp_st_enabled[pid, :], img_enabled=exp_img_enabled[pid])
        action = self.distribution.sample(agent_info)
        if self.debug:
            is_clipped = np.logical_or((action <= self.act_space.low), (action >= self.act_space.high))
            num_clips = np.count_nonzero(is_clipped)
            agent_info["clips"] = num_clips

        # logging.debug("tx a:{},\n".format(action))

        return action, agent_info

    def concat_paths(self, paths):
        state_input = concat([path["observation"][0] for path in paths])

        times = concat([path["times"] for path in paths], axis=0)
        returns = concat([path["return"] for path in paths])
        img_enabled = concat([path["img_enabled"] for path in paths])
        st_enabled = concat([path["st_enabled"] for path in paths])
        action_n = concat([path["action"] for path in paths])
        advant_n = concat([path["advantage"] for path in paths])
        action_dist_means_n = concat([path["mean"] for path in paths])
        action_dist_logstds_n = concat([path["log_std"] for path in paths])
        feed = {self.net.state_input: state_input,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n,

                self.critic.st_enabled: st_enabled,
                self.critic.img_enabled: img_enabled,
                self.net.st_enabled: st_enabled,
                self.net.img_enabled: img_enabled,
                }
        path_dict = {"state_input": state_input,
                     "times": times,
                     "returns": returns,
                     "img_enabled": img_enabled,
                     "st_enabled": st_enabled}
        if self.n_imgfeat > 0:
            img_input = concat([path["observation"][1] for path in paths])
            feed[self.net.img_input] = img_input
            feed[self.critic.img_input] = img_input
            path_dict["img_input"] = img_input
        return feed, path_dict

    def compute_critic(self, states):
        self.critic_lock.acquire_read()
        result = self.critic.predict(states)
        self.critic_lock.release_read()
        return result

    def check_batch_finished(self, time, epis):
        if self.batch_mode == "episode":
            return epis >= self.batch_size
        if self.batch_mode == "timestep":
            return time >= self.batch_size

    def compute_update(self, paths):

        # if self.separate_update:
        #     if self.n_update % 4< 2 :
        #         self.update_critic = True
        #         self.update_policy = False
        #     else:
        #         self.update_critic = False
        #         self.update_policy = True

        self.update_critic = True
        self.update_policy = True

        # if self.n_update % 2 != 0 or self.n_update < 20:
        #     self.update_critic = False
        #     self.update_policy = False
        #     self.critic.fit(path_dict, update_mode="img", num_pass=2)
        # else:
        #     self.update_critic = True
        #     self.update_policy = True
        feed, path_dict = self.concat_paths(paths)

        if self.n_update < self.n_pretrain:
            pass
            # self.saved_paths = [*self.saved_paths, *paths]
        elif self.n_update == self.n_pretrain:
            pass
            # pretrain_model_path = self.feat_saver.save(self.session, 'pretrain_model')
            # logging.debug("model_path: {}".format(pretrain_model_path))

            # self.saver.restore(self.session, self.init_model_path)
            # self.feat_saver.restore(self.session, 'pretrain_model')
            # self.update_critic = True
            # self.update_policy = True

            # _, img_path_dict = self.concat_paths(self.saved_paths)
            # self.saved_paths = []
            # self.critic.fit(img_path_dict, update_mode="img", num_pass=5)
            # self.saver.save(self.session, 'pretrained_model')
        self.critic_lock.acquire_write()
        if self.n_update < self.n_pretrain:
            self.critic.fit(path_dict, update_mode="img", num_pass=2)


        if self.update_critic:
            self.critic.fit(path_dict, update_mode="full", num_pass=10)
        self.critic_lock.release_write()
        self.n_update += 1

        # logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))
        advant_n = feed[self.net.advant]
        state_input = feed[self.net.state_input]
        action_dist_logstds_n = feed[self.net.old_dist_logstds_n]
        batch_size = advant_n.shape[0]
        thprev = self.get_flat_params()  # get theta_old
        if self.debug:
            # logging.debug("state max: {}\n min: {}".format(state_input.max(axis=0), state_input.min(axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
        if not self.update_policy:
            return None, None


        if self.mode == "ADA_KL":
            self.policy_lock.acquire_write()
            surr_o, kl_o, ent_o = run_batched(self.a_losses, feed, batch_size, self.session,
                                              minibatch_size=self.minibatch_size,
                                              extra_input={self.a_beta: self.a_beta_value})
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            # self.session.run(self.a_backup_op)
            early_stopped = False
            for e in range(self.update_per_epoch):
                grads = run_batched(self.a_grad_list, feed, batch_size, self.session,
                                    minibatch_size=self.minibatch_size,
                                    extra_input={self.a_beta: self.a_beta_value}
                                    )
                grad_dict = {p: v for (p, v) in zip(self.a_grad_placeholders, grads)}
                _ = self.session.run(self.a_apply_grad,
                                     feed_dict={**grad_dict,
                                                self.a_sym_step_size: self.a_step_size})
                kl_new = run_batched(self.a_losses[1], feed, batch_size, self.session,
                                     minibatch_size=self.minibatch_size,
                                     extra_input={self.a_beta: self.a_beta_value})
                if kl_new > self.target_kl * 4:
                    logging.debug("KL too large, early stop")
                    early_stopped = True
                    break

            surr_new, ent_new = run_batched([self.a_losses[0], self.a_losses[2]], feed, batch_size, self.session,
                                            minibatch_size=self.minibatch_size,
                                            extra_input={self.a_beta: self.a_beta_value})
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            if kl_new > self.target_kl * 2:
                self.a_beta_value = np.minimum(self.a_beta_max, 1.5 * self.a_beta_value)
                if self.a_beta_value > self.a_beta_max - 5 or early_stopped:
                    self.a_step_size = np.maximum(self.a_min_step_size, self.a_step_size / 1.5)
                logging.debug('beta -> %s' % self.a_beta_value)
                logging.debug('step_size -> %s' % self.a_step_size)
            elif kl_new < self.target_kl / 2:
                self.a_beta_value = np.maximum(self.a_beta_min, self.a_beta_value / 1.5)
                if self.a_beta_value < self.a_beta_min:
                    self.a_step_size = np.minimum(self.a_max_step_size, 1.5 * self.a_step_size)
                logging.debug('beta -> %s' % self.a_beta_value)
                logging.debug('step_size -> %s' % self.a_step_size)
            else:
                logging.debug('beta OK = %s' % self.a_beta_value)
                logging.debug('step_size OK = %s' % self.a_step_size)

            # if kl_new > self.target_kl * 30:
            #     logging.debug("KL too large, rollback")
            #     self.session.run(self.a_rollback_op)
            self.policy_lock.release_write()
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
