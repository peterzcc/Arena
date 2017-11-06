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
                 n_imgfeat=0,
                 target_kl=0.003,
                 minibatch_size=128,
                 gamma=0.995,
                 gae_lam=0.97,
                 mode="ADA_KL",
                 num_actors=1,
                 batch_size=20,
                 batch_mode="episode",
                 recompute_old_dist=False,
                 update_per_epoch=4):
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
        self.exp_st_enabled = None
        self.exp_img_enabled = None
        self.batch_mode = batch_mode  # "episode" #"timestep"
        self.batch_size = batch_size
        if self.batch_mode == "timestep":
            self.batch_barrier = thd.Barrier(num_actors)
        else:
            self.batch_barrier = None

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

        if self.mode == "SURP":
            self.net = MultiNetwork(scope="state_agent",
                                    observation_space=self.ob_space,
                                    action_shape=self.act_space.shape,
                                    n_imgfeat=0,
                                    extra_feaatures=[],
                                    conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)),  # (((3, 3), 2, 2),),  #
                                    comb_method=self.comb_method,
                                    min_std=min_std,
                                    distibution=self.distribution,
                                    session=self.session
                                    )
            self.target_net = MultiNetwork(scope="target_agent",
                                           observation_space=self.ob_space,
                                           action_shape=self.act_space.shape,
                                           n_imgfeat=self.n_imgfeat,
                                           extra_feaatures=[],
                                           conv_sizes=(((4, 4), 64, 2), ((3, 3), 64, 1)),
                                           comb_method=self.comb_method,
                                           min_std=min_std,
                                           distibution=self.distribution,
                                           session=self.session
                                           )
            self.executer_net = self.target_net
        else:
            self.net = MultiNetwork(scope="state_agent",
                                    observation_space=self.ob_space,
                                    action_shape=self.act_space.shape,
                                    n_imgfeat=self.n_imgfeat,
                                    extra_feaatures=[],
                                    # [],  #[np.zeros((4,), dtype=np.float32)],  #
                                    conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)),  # (((3, 3), 2, 2),),  #
                                    comb_method=self.comb_method,
                                    min_std=min_std,
                                    distibution=self.distribution,
                                    session=self.session
                                    )
            self.executer_net = self.net


        self.k_p_l2 = 0.01

        self.losses = self.net.losses
        var_list = self.net.var_list

        # if self.mode == "TRPO":
        #     self.flat_tangent = tf.placeholder(dtype, shape=[None])
        #     shapes = list(map(var_shape, var_list))
        #     start = 0
        #     tangents = []
        #     for shape in shapes:
        #         size = np.prod(shape)
        #         param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
        #         tangents.append(param)
        #         start += size
        #     self.gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        #     self.fvp = flatgrad(tf.reduce_sum(self.gvp), var_list)  # get kl''*p
        #     self.is_real_data = tf.placeholder(shape=(self.minibatch_size,), dtype=tf.float32)
        #
        #     grad_loglikelihood = flatgrad(self.new_likelihood_sym, var_list)
        #     list_logl = tf.unstack(self.new_likelihood_sym, num=self.minibatch_size)
        #     batch_grad = tf.stack([flatgrad(l, var_list) for l in list_logl]) * tf.expand_dims(self.is_real_data,
        #                                                                                        axis=1)
        #     batch_gT_x = tf.matmul(batch_grad, tf.expand_dims(self.flat_tangent, axis=1))
        #     self.batch_g_gT_x = tf.reshape(tf.matmul(batch_grad, batch_gT_x, transpose_a=True, transpose_b=False), [-1])
        #     gT_x = tf.reduce_sum(grad_loglikelihood * self.flat_tangent)
        #     self.g_gT_x = grad_loglikelihood * gT_x
        #
        self.target_kl = target_kl


        # adaptive kl
        self.a_beta = tf.placeholder(shape=[], dtype=tf.float32, name="a_beta")
        self.a_beta_value = 3
        self.a_eta_value = 50
        self.a_surr = -tf.reduce_mean(self.net.raw_surr) + self.a_beta * self.net.kl + \
                      self.a_eta_value * tf.square(tf.maximum(0.0, self.net.kl - 2.0 * self.target_kl))
        self.a_step_size = 0.0001
        self.a_max_step_size = 0.01
        self.a_min_step_size = 1e-7
        self.a_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="a_step_size")
        self.a_opt = tf.train.AdamOptimizer(learning_rate=self.a_sym_step_size)
        self.a_grad_list = tf.gradients(self.a_surr, var_list)
        self.a_grad_placeholders = [tf.placeholder(tf.float32, shape=g.shape, name="a_grad_sym")
                                    for g in self.a_grad_list]
        self.a_old_parameters = [tf.Variable(tf.zeros(v.shape, dtype=tf.float32), name="a_old_param") for v in var_list]
        self.a_backup_op = [tf.assign(old_v, v) for (old_v, v) in zip(self.a_old_parameters, var_list)]
        self.a_rollback_op = [[tf.assign(v, old_v) for (old_v, v) in zip(self.a_old_parameters, var_list)]]
        self.a_apply_grad = self.a_opt.apply_gradients(grads_and_vars=zip(self.a_grad_placeholders, var_list))
        self.a_losses = [self.a_surr, self.net.kl, self.net.ent]
        self.a_beta_max = 35.0
        self.a_beta_min = 1.0 / 35.0
        self.update_per_epoch = update_per_epoch


        self.saved_paths = []

        # self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)

        if len(self.critic.img_var_list) > 0:
            self.feat_saver = tf.train.Saver(var_list=self.critic.img_var_list)
        self.saver = tf.train.Saver(var_list=self.net.var_list)
        if self.mode == "SURP":
            self.regression_loss = tf.reduce_mean(
                tf.square(self.net.action_dist_means_n - self.target_net.action_dist_means_n))
            self.s_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.s_train_op = self.s_opt.minimize(self.regression_loss, var_list=self.target_net.var_list)

        # self.init_model_path = self.saver.save(self.session, 'init_model')
        self.n_update = 0
        self.n_pretrain = 0
        self.separate_update = True
        self.update_critic = False
        self.update_policy = True
        self.debug = True
        self.recompute_old_dist = recompute_old_dist
        self.session.run(tf.global_variables_initializer())
        self.restore_from_pretrained_policy = True
        if self.restore_from_pretrained_policy:
            self.saver.restore(self.session, 'policy_parameter')
        self.session.run(tf.assign(self.target_net.log_vars, self.net.log_vars), feed_dict={})

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


        all_st_enabled = False
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
            self.exp_st_enabled = np.expand_dims(st_enabled, 0)
            self.exp_img_enabled = img_enabled
        else:
            self.obs_list[pid] = observation
            self.execution_barrier.wait()
            if pid == 0:
                obs = [np.stack([ob[0] for ob in self.obs_list])]
                if self.n_imgfeat > 0:
                    obs.append(np.stack([o[1] for o in self.obs_list]))
                st_enabled, img_enabled = self.get_state_activation(self.n_update)
                st_enabled = np.tile(st_enabled, (self.num_actors, 1))
                img_enabled = np.repeat(img_enabled, self.num_actors, axis=0)
                self.exp_st_enabled = st_enabled
                self.exp_img_enabled = img_enabled

        if pid == 0:
            feed = {self.executer_net.state_input: obs[0],
                    self.critic.st_enabled: self.exp_st_enabled,
                    self.critic.img_enabled: self.exp_img_enabled,
                    self.executer_net.st_enabled: self.exp_st_enabled,
                    self.executer_net.img_enabled: self.exp_img_enabled,
                    }
            if self.n_imgfeat > 0:
                feed[self.critic.img_input] = obs[1]
                feed[self.executer_net.img_input] = obs[1]
            self.policy_lock.acquire_read()
            action_dist_means_n, action_dist_log_stds_n = \
                self.session.run([self.executer_net.action_dist_means_n, self.executer_net.action_dist_log_stds_n],
                                 feed)
            self.act_means = action_dist_means_n
            self.act_logstds = action_dist_log_stds_n
            self.policy_lock.release_read()
        self.execution_barrier.wait()

        # logging.debug("am:{},\nastd:{}".format(action_dist_means_n[0],action_dist_stds_n[0]))

        agent_info = dict(mean=self.act_means[pid, :], log_std=self.act_logstds,
                          st_enabled=self.exp_st_enabled[pid, :], img_enabled=self.exp_img_enabled[pid])
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

        feed = {self.net.state_input: state_input,
                self.net.advant: advant_n,

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
        img_input = None
        if self.n_imgfeat > 0:
            img_input = concat([path["observation"][1] for path in paths])
            feed[self.net.img_input] = img_input
            feed[self.critic.img_input] = img_input
            path_dict["img_input"] = img_input
        if self.recompute_old_dist:
            feed_forw = {self.net.state_input: state_input,
                         self.critic.st_enabled: st_enabled,
                         self.critic.img_enabled: img_enabled,
                         self.net.st_enabled: st_enabled,
                         self.net.img_enabled: img_enabled,
                         }
            if self.n_imgfeat > 0:
                feed_forw[self.critic.img_input] = img_input
                feed_forw[self.net.img_input] = img_input
            action_dist_means_n, action_dist_logstds = \
                self.session.run([self.net.action_dist_means_n, self.net.action_dist_log_stds_n],
                                 feed_forw)
            action_dist_logstds_n = np.tile(action_dist_logstds, (state_input.shape[0], 1))
            feed.update(
                {
                    self.net.old_dist_means_n: action_dist_means_n,
                    self.net.old_dist_logstds_n: action_dist_logstds_n,
                }
            )
        else:
            action_dist_means_n = concat([path["mean"] for path in paths])
            action_dist_logstds_n = concat([path["log_std"] for path in paths])
            feed.update(
                {
                    self.net.old_dist_means_n: action_dist_means_n,
                    self.net.old_dist_logstds_n: action_dist_logstds_n,
                }
            )
        return feed, path_dict

    def compute_critic(self, states):
        self.critic_lock.acquire_read()
        result = self.critic.predict(states)
        self.critic_lock.release_read()
        return result

    def check_batch_finished(self, time, epis):
        if self.batch_mode == "episode":
            assert epis <= self.batch_size
            return epis == self.batch_size
        if self.batch_mode == "timestep":
            assert time <= self.batch_size
            return time == self.batch_size

    def supervised_update(self, paths):
        state_input = concat([path["observation"][0] for path in paths])

        img_enabled = concat([path["img_enabled"] for path in paths])
        st_enabled = concat([path["st_enabled"] for path in paths])
        action_n = concat([path["action"] for path in paths])
        advant_n = concat([path["advantage"] for path in paths])

        feed = {self.net.state_input: state_input,
                self.net.advant: advant_n,
                self.net.action_n: action_n,
                self.net.st_enabled: st_enabled,
                self.net.img_enabled: img_enabled,
                self.target_net.state_input: state_input,
                self.target_net.advant: advant_n,
                self.target_net.action_n: action_n,
                self.target_net.st_enabled: st_enabled,
                self.target_net.img_enabled: img_enabled
                }

        img_input = None
        action_dist_logstds_n = concat([path["log_std"] for path in paths])
        if self.n_imgfeat > 0:
            img_input = concat([path["observation"][1] for path in paths])
            feed[self.target_net.img_input] = img_input
        batch_size = advant_n.shape[0]
        if self.debug:
            # logging.debug("state max: {}\n min: {}".format(state_input.max(axis=0), state_input.min(axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
        loss_o = run_batched(self.regression_loss, feed, batch_size, self.session,
                             minibatch_size=self.minibatch_size)
        logging.debug("\nold reg loss: {}\n".format(loss_o))
        for n_pass in range(self.update_per_epoch):
            training_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):  # TODO: verify this
                if start > batch_size - 2 * self.minibatch_size:
                    end = batch_size
                else:
                    end = start + self.minibatch_size
                slc = training_inds[range(start, end)]
                this_feed = {k: v[slc] for (k, v) in list(feed.items())}
                self.session.run(self.s_train_op, feed_dict=this_feed)
                if end == batch_size:
                    break

        loss_n = run_batched(self.regression_loss, feed, batch_size, self.session,
                             minibatch_size=self.minibatch_size)
        logging.debug("\nnew reg loss: {}\n".format(loss_n))


    def compute_update(self, paths):

        # if self.separate_update:
        #     if self.n_update % 4< 2 :
        #         self.update_critic = True
        #         self.update_policy = False
        #     else:
        #         self.update_critic = False
        #         self.update_policy = True


        # if self.n_update % 2 != 0 or self.n_update < 20:
        #     self.update_critic = False
        #     self.update_policy = False
        #     self.critic.fit(path_dict, update_mode="img", num_pass=2)
        # else:
        #     self.update_critic = True
        #     self.update_policy = True
        if self.mode == "SURP":
            if self.update_policy:
                self.policy_lock.acquire_write()
                self.supervised_update(paths)
                self.policy_lock.release_write()
            return None, None
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

        # if self.n_update % 20 == 0:
        #     self.saver.save(self.session,'policy_parameter')
        self.critic_lock.acquire_write()
        if self.n_update < self.n_pretrain:
            self.critic.fit(path_dict, update_mode="img", num_pass=2)


        if self.update_critic:
            self.critic.fit(path_dict, update_mode="full", num_pass=self.update_per_epoch)
        self.critic_lock.release_write()
        self.n_update += 1

        # logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))
        advant_n = feed[self.net.advant]
        state_input = feed[self.net.state_input]
        action_dist_logstds_n = feed[self.net.old_dist_logstds_n]
        batch_size = advant_n.shape[0]

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

            # if self.mode=="TRPO":
            #     thprev = self.get_flat_params()  # get theta_old
            #     if self.use_empirical_fim:
            #         def fisher_vector_product(p):
            #             # feed[self.flat_tangent] = p
            #
            #             fvp = self.run_batched_fvp(self.batch_g_gT_x, feed, batch_size, self.session,
            #                                        minibatch_size=self.minibatch_size,
            #                                        extra_input={self.flat_tangent: p})
            #             return fvp + self.cg_damping * p
            #     else:
            #         def fisher_vector_product(p):
            #             # feed[self.flat_tangent] = p
            #
            #             fvp = run_batched(self.fvp, feed, batch_size, self.session, minibatch_size=self.minibatch_size,
            #                               extra_input={self.flat_tangent: p})
            #             return fvp + self.cg_damping * p
            #
            #     g = run_batched(self.pg, feed, batch_size, self.session, minibatch_size=self.minibatch_size)
            #
            #     stepdir = cg(fisher_vector_product, -g, cg_iters=self.cg_iters, residual_tol=1e-5)
            #
            #     sAs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
            #     # if shs<0, then the nan error would appear
            #     lm = np.sqrt(sAs / self.max_kl)
            #     fullstep = stepdir / lm
            #     neggdotstepdir = -g.dot(stepdir)
            #     logging.debug("\nlagrange multiplier:{}\tgnorm:{}\t".format(lm, np.linalg.norm(g)))
            #
            #     def loss(th):
            #         self.set_params_with_flat_data(th)
            #         surr = run_batched(self.surr, feed, batch_size, self.session, minibatch_size=self.minibatch_size)
            #         return surr
            #
            #     if self.debug:
            #         surr_o, kl_o, ent_o = run_batched(self.losses, feed, batch_size, self.session,
            #                                           minibatch_size=self.minibatch_size)
            #         logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            #         # logging.debug("\nold theta: {}\n".format(np.linalg.norm(thprev)))
            #     logging.debug("\nfullstep: {}\n".format(np.linalg.norm(fullstep)))
            #     theta, d_theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
            #     self.set_params_with_flat_data(theta)
            #
            #     if self.debug:
            #         surr_new, kl_new, ent_new = run_batched(self.losses, feed, batch_size, self.session,
            #                                                 minibatch_size=self.minibatch_size)
            #         logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            #         # logging.debug("\nnew theta: {}\nd_theta: {}\n".format(np.linalg.norm(theta), np.linalg.norm(d_theta)))
            #     return None, None

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
