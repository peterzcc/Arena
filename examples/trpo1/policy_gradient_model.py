from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st, explained_variance_batched, tf_run_batched, batch_run_forward
# from baseline import Baseline
from multi_baseline import MultiBaseline
from prob_types import DiagonalGaussian, Categorical, CategoricalWithProb
from network_models import MultiNetwork
import numpy as np
import random
import threading as thd
import logging
from dict_memory import DictMemory
from read_write_lock import ReadWriteLock
from cnn import ConvAutoencorder, cnn_network, ConvFcAutoencorder
# from baselines import common
# from baselines.common import tf_util as U
from baselines.acktr import kfac
from arena.experiment import Experiment
from scaling_orth import ScalingOrth
concat = np.concatenate
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32


class PolicyGradientModel(ModelWithCritic):
    def __init__(self, observation_space, action_space,
                 name="agent",
                 session=None,
                 conv_sizes=(((3, 3), 16, 2), ((3, 3), 16, 2), ((3, 3), 4, 2)),
                 min_std=1e-6,
                 timestep_limit=1000,
                 n_imgfeat=0,
                 f_target_kl=None,
                 lr=0.0001,
                 critic_lr=0.0003,
                 minibatch_size=128,
                 mode="ACKTR",
                 surr_loss="PPO",
                 num_actors=1,
                 f_batch_size=None,
                 batch_mode="episode",
                 update_per_epoch=4,
                 kl_history_length=1,
                 ent_k=0,
                 comb_method=concat_feature,
                 load_old_model=False,
                 model_load_dir="./model",
                 should_train=True,
                 f_train_this_epoch=lambda x: True,
                 parallel_predict=True,
                 save_model=10,
                 loss_type="PPO",
                 max_grad_norm=0.5,
                 is_switcher_with_init_len=0,
                 is_decider=False,
                 reset_exp=False,
                 const_action=None
                 ):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space
        # logging.debug("\naction space:".format(action_space))
        # logging.debug("\nstate space: {} to {}".format(observation_space[0].low, observation_space[0].high))
        args = locals()
        logging.debug("model args:\n {}".format(args))

        # store constants
        self.min_std = min_std
        self.minibatch_size = minibatch_size
        self.use_empirical_fim = True
        self.mode = mode
        self.save_model = save_model
        self.best_mean_reward = - np.inf
        self.last_save = - np.inf
        self.policy_lock = ReadWriteLock()
        self.critic_lock = ReadWriteLock()
        self.num_actors = num_actors
        self.execution_barrier = thd.Barrier(num_actors)
        # self.act_means_p = None
        # self.act_logstds_p = None
        self.stored_dist_infos = None
        self.stored_actions = None
        self.obs_list = [None for i in range(num_actors)]
        self.act_list = [None for i in range(num_actors)]
        self.batch_ends = np.zeros(shape=(num_actors,), dtype=np.bool)
        self.exp_st_enabled = None
        self.exp_img_enabled = None
        self.batch_mode = batch_mode  # "episode" #"timestep"
        self.min_batch_length = 1000
        self.f_batch_size = f_batch_size
        self.batch_size = None if self.f_batch_size is None else self.f_batch_size(0)
        self.f_target_kl = f_target_kl
        self.kl_history_length = kl_history_length
        self.is_switcher_with_init_len = is_switcher_with_init_len
        if self.batch_mode == "timestep":
            self.batch_barrier = thd.Barrier(num_actors)
        else:
            self.batch_barrier = thd.Barrier(num_actors)

        gpu_options = tf.GPUOptions(allow_growth=True)  # False,per_process_gpu_memory_fraction=0.75)
        self.session = session if session is not None else tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                                                            log_device_placement=False))

        self.n_imgfeat = n_imgfeat if n_imgfeat is not None else self.ob_space[0].shape[0]
        self.comb_method = comb_method  # aggregate_feature#

        cnn_trainable = True
        if n_imgfeat < 0:
            cnn_fc_feat = None
        else:
            cnn_fc_feat = (64, n_imgfeat,)

        def f_build_img_net(t_input):
            return cnn_network(t_input, conv_sizes, cnn_activation=tf.nn.leaky_relu,
                               fc_sizes=cnn_fc_feat, fc_activation=tf.nn.leaky_relu,
                               initializer_fn=ScalingOrth)

        self.name = name if not self.is_switcher_with_init_len else name + "_switcher"
        self.is_decider = is_decider

        self.critic = MultiBaseline(session=self.session, observation_space=self.ob_space,
                                    minibatch_size=64,
                                    main_scope=self.name + "_critic",
                                    timestep_limit=timestep_limit,
                                    activation=tf.nn.leaky_relu,
                                    n_imgfeat=self.n_imgfeat,
                                    comb_method=self.comb_method,
                                    cnn_trainable=cnn_trainable,
                                    f_build_cnn=f_build_img_net,
                                    is_switcher_with_init_len=is_switcher_with_init_len,
                                    lr=critic_lr)
        if hasattr(self.act_space, "low"):
            self.distribution = DiagonalGaussian(dim=self.act_space.low.shape[0])
        else:
            self.distribution = Categorical(num_cat=self.act_space.n)

        self.theta = None
        # self.info_shape = dict(mean=self.act_space.shape,
        #                        logstd=self.act_space.shape,
        #                        clips=(),
        #                        img_enabled=(),
        #                        st_enabled=self.ob_space[0].low.shape)

        self.policy = MultiNetwork(scope=self.name + "_policy",
                                   observation_space=self.ob_space,
                                   action_space=self.act_space,
                                   n_imgfeat=self.n_imgfeat,
                                   extra_feaatures=[],
                                   comb_method=self.comb_method,
                                   min_std=min_std,
                                   distibution=self.distribution,
                                   session=self.session,
                                   cnn_trainable=cnn_trainable,
                                   f_build_cnn=f_build_img_net,
                                   is_switcher_with_init_len=is_switcher_with_init_len
                                   )
        self.executer_net = self.policy

        self.losses = self.policy.losses
        var_list = self.policy.var_list

        self.target_kl_initial = None if f_target_kl is None else f_target_kl(0)
        self.target_kl_sym = tf.placeholder(shape=[], dtype=tf.float32, name="target_kl")
        self.ent_k = ent_k
        self.ent_loss = 0 if self.ent_k == 0 else -self.ent_k * self.policy.ent
        self.fit_policy = None
        self.loss_type = loss_type
        self.rl_loss = (
            self.policy.ppo_surr if self.loss_type == "PPO" else self.policy.trad_surr_loss)
        if should_train:
            if self.mode == "ADA_KL":
                # adaptive kl
                self.a_beta = tf.placeholder(shape=[], dtype=tf.float32, name="a_beta")
                self.a_beta_value = 3
                self.a_eta_value = 50
                self.a_rl_loss = -tf.reduce_mean(self.policy.surr_n)
                self.a_kl_loss = self.a_beta * self.policy.kl + \
                                 self.a_eta_value * tf.square(
                                     tf.maximum(0.0, self.policy.kl - 2.0 * self.target_kl_sym))
                self.a_surr = self.a_rl_loss + self.a_kl_loss + self.ent_loss
                self.a_step_size = 0.0001
                self.a_max_step_size = 0.1
                self.a_min_step_size = 1e-7
                self.a_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="a_step_size")
                self.a_opt = tf.train.AdamOptimizer(learning_rate=self.a_sym_step_size)
                self.a_grad_list = tf.gradients(self.a_surr, var_list)
                self.a_rl_grad = tf.gradients(self.a_rl_loss, var_list)
                self.a_kl_grad = tf.gradients(self.a_kl_loss, var_list)

                self.hist_obs0 = []
                self.hist_obs1 = []
                self.hist_st_en = []
                self.hist_img_en = []
                self.a_grad_placeholders = [tf.placeholder(tf.float32, shape=g.shape, name="a_grad_sym")
                                            for g in self.a_grad_list]
                self.a_old_parameters = [tf.Variable(tf.zeros(v.shape, dtype=tf.float32), name="a_old_param") for v in
                                         var_list]
                self.a_backup_op = [tf.assign(old_v, v) for (old_v, v) in zip(self.a_old_parameters, var_list)]
                self.a_rollback_op = [[tf.assign(v, old_v) for (old_v, v) in zip(self.a_old_parameters, var_list)]]
                self.a_apply_grad = self.a_opt.apply_gradients(grads_and_vars=zip(self.a_grad_placeholders, var_list))
                self.a_losses = [self.a_rl_loss, self.policy.kl, self.policy.ent]
                self.a_beta_max = 35.0
                self.a_beta_min = 1.0 / 35.0
                self.update_per_epoch = update_per_epoch
                self.fit_policy = self.fit_adakl
            elif self.mode == "ACKTR":
                self.k_stepsize = tf.Variable(initial_value=np.float32(0.03), name='stepsize')
                self.k_momentum = 0.9
                self.k_optim = kfac.KfacOptimizer(learning_rate=self.k_stepsize,
                                                  cold_lr= self.k_stepsize * (1 - self.k_momentum),
                                                  momentum=self.k_momentum,
                                                  kfac_update=2,
                                                  epsilon=1e-2, stats_decay=0.99,
                                                  async=True, cold_iter=1,
                                                  weight_decay_dict={}, max_grad_norm=None)

                self.k_surr_loss = self.policy.ppo_surr if self.loss_type == "PPO" else self.policy.trad_surr_loss
                self.k_final_loss = self.k_surr_loss + self.ent_loss

                # self.k_grads = tf.gradients(self.k_final_loss, var_list)
                # self.k_grads_ph = [tf.placeholder(tf.float32, shape=v.shape, name="k_grad_sym")
                #                    for v in self.policy.var_list]
                # k_grads_ph_var = [ (g, v) for g,v in zip(self.k_grads_ph, self.policy.var_list) ]
                # self.k_apply_grad, self.k_q_runner = self.k_optim.apply_stats_and_grads(
                #     grads=k_grads_ph_var,
                #     loss_sampled=self.policy.mean_loglike,
                #     var_list=self.policy.var_list)

                self.k_update_op, self.k_q_runner = self.k_optim.minimize(self.k_final_loss,
                                                                          self.policy.mean_loglike,
                                                                          var_list=self.policy.var_list)
                self.k_enqueue_threads = []
                self.k_coord = tf.train.Coordinator()
                self.fit_policy = self.fit_acktr
            elif self.mode == "PG":
                with tf.variable_scope("pg") as scope:
                    self.p_step_size = lr
                    self.p_max_step_size = 0.1
                    self.p_min_step_size = 1e-7
                    self.p_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="p_step_size")
                    self.pg_optim = tf.train.AdamOptimizer(learning_rate=self.p_sym_step_size)
                    self.p_beta = tf.placeholder(shape=[], dtype=tf.float32, name="a_beta")
                    self.p_beta_max = 35.0
                    self.p_beta_min = 1.0 / 35.0
                    self.p_beta_value = 3
                    self.p_eta_value = 50
                    if self.f_target_kl is not None:
                        self.p_kl_loss = self.p_beta * self.policy.kl + \
                                         self.p_eta_value * tf.square(
                            tf.maximum(0.0, self.policy.kl - 2.0 * self.target_kl_sym))
                    else:
                        self.p_kl_loss = 0.0
                    self.pg_loss = self.rl_loss + self.ent_loss + self.p_kl_loss

                    self.p_grads = tf.gradients(self.pg_loss, self.policy.var_list)

                    self.p_grads_sum = [tf.Variable(tf.zeros(g.shape), dtype=g.dtype, trainable=False)
                                        for g in self.p_grads]
                    self.p_accum_reset = [tf.assign(p, tf.zeros(p.shape, dtype=p.dtype)) for p in self.p_grads_sum]
                    self.p_accum_op = [tf.assign_add(p, g) for (p, g) in zip(self.p_grads_sum, self.p_grads)]

                    if max_grad_norm is not None:
                        self.p_final_grads, self.g_norm = tf.clip_by_global_norm(self.p_grads_sum, max_grad_norm)
                    else:
                        self.p_final_grads = self.p_grads_sum
                    p_grads_var = [(g, v) for g, v in zip(self.p_grads_sum, self.policy.var_list)]
                    self.p_apply_grad = self.pg_optim.apply_gradients(p_grads_var)

                self.fit_policy = self.fit_pg
            else:
                raise NotImplementedError

        # self.saved_paths = []

        # self.summary_writer = tf.summary.FileWriter('./summary', self.session.graph)


        # self.init_model_path = self.saver.save(self.session, 'init_model')
        self.n_update = 0
        self.n_fit = 0
        self.n_pretrain = 0
        self.separate_update = True
        self.should_update_critic = True
        self.should_update_policy = True
        self.should_train = should_train
        self.f_train_this_epoch = f_train_this_epoch
        self.debug = True
        self.recompute_old_dist = False
        self.session.run(tf.global_variables_initializer())
        if self.mode == "ACKTR" and self.should_train:
            for qr in [self.k_q_runner]:
                if (qr != None):
                    self.k_enqueue_threads.extend(qr.create_threads(self.session, coord=self.k_coord, start=True))
        self.model_load_path = model_load_dir + "/" + self.name
        self.model_save_path = "./" + Experiment.EXP_NAME + "/" + self.name
        self.full_model_saver = tf.train.Saver(
            var_list=[*self.critic.var_list, *self.critic.var_notrain, *self.policy.var_list])
        self.has_loaded_model = False
        self.load_old_model = load_old_model
        self.should_reset_exp = reset_exp
        self.parallel_predict = parallel_predict
        self.has_reset_fix_ter_weight = False
        self.const_action = const_action

        # if self.load_old_model and not self.has_loaded_model:
        #     self.restore_parameters()

    def get_state_activation(self, t_batch):
        if self.comb_method != aggregate_feature:
            all_st_enabled = True
            st_enabled = np.ones(self.ob_space[0].shape) if all_st_enabled else np.zeros(self.ob_space[0].shape)
            img_enabled = np.array((1.0,))
        else:
            all_st_enabled = True
            st_enabled = np.ones(self.ob_space[0].shape) if all_st_enabled else np.zeros(self.ob_space[0].shape)
            img_enabled = np.array((1.0 - all_st_enabled,))
        return st_enabled, img_enabled

    def restore_parameters(self):
        logging.debug("Loading {}".format(self.model_load_path))
        self.full_model_saver.restore(self.session, self.model_load_path)
        self.has_loaded_model = True

    def predict(self, observation, pid=0):
        if self.const_action is not None:
            return self.const_action, {}
        if self.load_old_model and not self.has_loaded_model:
            self.policy_lock.acquire_write()
            if not self.has_loaded_model:
                self.restore_parameters()
                if self.should_reset_exp:
                    logging.info("reset exploration")
                    self.session.run(self.executer_net.reset_exp)
            self.policy_lock.release_write()
        # if self.should_train and self.is_switcher_with_init_len and not self.has_reset_fix_ter_weight \
        #         and self.n_fit == 0 \
        #         and self.f_train_this_epoch(self.n_update):
        #     self.policy_lock.acquire_write()
        #     if not self.has_reset_fix_ter_weight:
        #         self.session.run(tf.assign(self.executer_net.fixed_ter_weight, 0))
        #         self.has_reset_fix_ter_weight = True
        #     self.policy_lock.release_write()
        if self.parallel_predict:
            obs = None
            if self.num_actors == 1:
                if len(observation[0].shape) == len(self.ob_space[0].shape):
                    obs = [np.expand_dims(observation[0], 0)]
                else:
                    obs = observation
                if self.n_imgfeat != 0:
                    obs.append(np.expand_dims(observation[1], 0))
                if self.is_switcher_with_init_len:
                    obs.append(np.expand_dims(observation[-1], 0))
                st_enabled, img_enabled = self.get_state_activation(self.n_update)
                self.exp_st_enabled = np.expand_dims(st_enabled, 0)
                self.exp_img_enabled = img_enabled
            else:
                self.obs_list[pid] = observation
                self.execution_barrier.wait()
                if pid == 0:
                    obs = [np.stack([ob[0] for ob in self.obs_list])]
                    if self.n_imgfeat != 0:
                        obs.append(np.stack([o[1] for o in self.obs_list]))
                    if self.is_switcher_with_init_len:
                        obs.append(np.stack([o[2] for o in self.obs_list]))
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
                if self.n_imgfeat != 0:
                    feed[self.critic.img_input] = obs[1]
                    feed[self.executer_net.img_input] = obs[1]
                if self.is_switcher_with_init_len:
                    feed[self.critic.hrl_meta_input] = obs[2]
                    feed[self.executer_net.hrl_meta_input] = obs[2]
                self.policy_lock.acquire_read()
                # self.dist_infos = \
                #     self.session.run(self.executer_net.new_dist_info_vars,
                #                      feed)
                self.dist_infos, self.stored_actions = self.session.run(
                    [self.executer_net.dist_vars, self.executer_net.sampled_action],
                    feed)
                self.policy_lock.release_read()
            self.execution_barrier.wait()
            info_this_thread = {k: v[pid, :] for k, v in self.dist_infos.items()}
            agent_info = {**info_this_thread,
                          **dict(st_enabled=self.exp_st_enabled[pid, :], img_enabled=self.exp_img_enabled[pid])}
            action = np.take(self.stored_actions, pid, axis=0)
        else:
            if len(observation[0].shape) == len(self.ob_space[0].shape):
                obs = [np.expand_dims(observation[0], 0)]
            else:
                obs = observation

            st_enabled, img_enabled = self.get_state_activation(self.n_update)
            self.exp_st_enabled = np.expand_dims(st_enabled, 0)
            self.exp_img_enabled = img_enabled

            feed = {self.executer_net.state_input: obs[0],
                    self.critic.st_enabled: self.exp_st_enabled,
                    self.critic.img_enabled: self.exp_img_enabled,
                    self.executer_net.st_enabled: self.exp_st_enabled,
                    self.executer_net.img_enabled: self.exp_img_enabled,
                    }
            if self.n_imgfeat != 0:
                img_input = np.expand_dims(observation[1], 0)
                feed[self.critic.img_input] = img_input
                feed[self.executer_net.img_input] = img_input
            if self.is_switcher_with_init_len:
                meta_input = np.expand_dims(observation[-1], 0)
                feed[self.policy.hrl_meta_input] = meta_input
                feed[self.critic.hrl_meta_input] = meta_input

            self.policy_lock.acquire_read()
            self.dist_infos, self.stored_actions = self.session.run(
                [self.executer_net.dist_vars, self.executer_net.sampled_action, ],
                feed)
            self.policy_lock.release_read()

            info_this_thread = {k: v[0, :] for k, v in self.dist_infos.items()}
            agent_info = {**info_this_thread,
                          **dict(st_enabled=self.exp_st_enabled[0, :],
                                 img_enabled=self.exp_img_enabled[0])}

            action = np.take(self.stored_actions, 0, axis=0)

        if self.debug:
            if hasattr(self.act_space, 'low'):
                is_clipped = np.logical_or((action <= self.act_space.low), (action >= self.act_space.high))
                num_clips = np.count_nonzero(is_clipped)
                agent_info["clips"] = num_clips

        return action, agent_info

    def concat_paths(self, paths):
        state_input = paths["observation"][0]  # concat([path["observation"][0] for path in paths])

        times = paths["times"]
        returns = paths["return"]  # concat([path["return"] for path in paths])
        img_enabled = paths["img_enabled"]  # concat([path["img_enabled"] for path in paths])
        st_enabled = paths["st_enabled"]  # concat([path["st_enabled"] for path in paths])
        action_n = paths["action"]
        advant_n = paths["advantage"]
        rewards = paths["reward"]
        dist_vars = {}
        for k in self.policy.dist_vars.keys():
            dist_vars[self.policy.old_vars[k]] = paths[k]  # concat([path[k] for path in paths])

        feed = {self.policy.state_input: state_input,
                self.policy.advant: advant_n,
                self.policy.action_n: action_n,
                self.critic.st_enabled: st_enabled,
                self.critic.img_enabled: img_enabled,
                self.policy.st_enabled: st_enabled,
                self.policy.img_enabled: img_enabled,
                }

        feed_critic = {self.critic.state_input: state_input,
                       self.critic.time_input: times, self.critic.y: returns,
                       self.critic.st_enabled: st_enabled, self.critic.img_enabled: img_enabled}

        if self.n_imgfeat != 0:
            img_input = paths["observation"][1]  # concat([path["observation"][1] for path in paths])
            feed[self.policy.img_input] = img_input
            feed[self.critic.img_input] = img_input
            feed_critic[self.critic.img_input] = img_input
        if self.is_switcher_with_init_len:
            hrl_meta_input = paths["observation"][-1]  # concat([path["observation"][2] for path in paths])
            feed[self.policy.hrl_meta_input] = hrl_meta_input
            feed[self.critic.hrl_meta_input] = hrl_meta_input
            feed_critic[self.critic.hrl_meta_input] = hrl_meta_input
        # action_dist_means_n = concat([path["mean"] for path in aggre_paths])
        # action_dist_logstds_n = concat([path["log_std"] for path in aggre_paths])

        feed.update({**dist_vars})
        no_ter_train = False
        if no_ter_train and self.is_switcher_with_init_len:
            is_root_decision = action_n != 0
            feed = {k: v[is_root_decision] for (k, v) in feed.items()}

        extra = {"rewards": rewards}

        return feed, feed_critic, extra

    def compute_critic(self, states):
        self.critic_lock.acquire_read()
        result = self.critic.predict(states)
        self.critic_lock.release_read()
        return result

    def check_batch_finished(self, time, epis):
        if self.batch_mode == "episode":
            return epis >= self.batch_size and time >= self.min_batch_length
        if self.batch_mode == "timestep":
            # if not time <= self.batch_size:
                # logging.debug("time: {} \nbatchsize: {}".format(time, self.batch_size))
                # assert time <= self.batch_size
            return time >= self.batch_size

    def increment_n_update(self):
        self.n_update += 1
        self.batch_size = self.f_batch_size(self.n_update)

    def handle_model_saving(self, mean_t_reward):
        if self.save_model > 0 and self.n_update > self.last_save + self.save_model:
            if mean_t_reward > self.best_mean_reward:
                self.best_mean_reward = mean_t_reward
                logging.debug("Saving {} with averew/step: {}".format(self.model_save_path, self.best_mean_reward))
                self.full_model_saver.save(self.session, self.model_save_path, write_state=False)
                self.last_save = self.n_update

    def fit_adakl(self, feed, num_samples, pid=None):
        target_kl_value = self.f_target_kl(self.n_update)
        logging.debug("\nbatch_size: {}\nkl_target: {}\n".format(self.batch_size, target_kl_value))
        surr_o, kl_o, ent_o = run_batched(self.a_losses, feed, num_samples, self.session,
                                          minibatch_size=self.minibatch_size,
                                          extra_input={self.a_beta: self.a_beta_value,
                                                       self.target_kl_sym: target_kl_value})
        logging.debug("\nold ppo_surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
        # self.session.run(self.a_backup_op)
        early_stopped = False
        kl_new = -1
        for e in range(self.update_per_epoch):
            grads = run_batched(self.a_grad_list, feed, num_samples, self.session,
                                minibatch_size=self.minibatch_size,
                                extra_input={self.a_beta: self.a_beta_value,
                                             self.target_kl_sym: target_kl_value}
                                )
            grad_dict = {p: v for (p, v) in zip(self.a_grad_placeholders, grads)}
            _ = self.session.run(self.a_apply_grad,
                                 feed_dict={**grad_dict,
                                            self.a_sym_step_size: self.a_step_size})
            kl_new = run_batched(self.a_losses[1], feed, num_samples, self.session,
                                 minibatch_size=self.minibatch_size,
                                 extra_input={self.a_beta: self.a_beta_value,
                                              self.target_kl_sym: target_kl_value})
            if kl_new > target_kl_value * 4:
                logging.debug("KL too large, early stop")
                early_stopped = True
                break

        surr_new, ent_new = run_batched([self.a_losses[0], self.a_losses[2]], feed, num_samples, self.session,
                                        minibatch_size=self.minibatch_size,
                                        extra_input={self.a_beta: self.a_beta_value,
                                                     self.target_kl_sym: target_kl_value})
        logging.debug("\nnew ppo_surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
        if kl_new > target_kl_value * 2:
            self.a_beta_value = np.minimum(self.a_beta_max, 1.5 * self.a_beta_value)
            if self.a_beta_value > self.a_beta_max / 1.2 or early_stopped:
                self.a_step_size = np.maximum(self.a_min_step_size, self.a_step_size / 1.5)
            logging.debug('beta -> %s' % self.a_beta_value)
            logging.debug('step_size -> %s' % self.a_step_size)
        elif kl_new < target_kl_value / 2:
            self.a_beta_value = np.maximum(self.a_beta_min, self.a_beta_value / 1.5)
            if self.a_beta_value < 1.2 * self.a_beta_min:
                self.a_step_size = np.minimum(self.a_max_step_size, 1.5 * self.a_step_size)
            logging.debug('beta -> %s' % self.a_beta_value)
            logging.debug('step_size -> %s' % self.a_step_size)
        else:
            logging.debug('beta OK = %s' % self.a_beta_value)
            logging.debug('step_size OK = %s' % self.a_step_size)

    def fit_acktr(self, feed, num_samples, pid=None):
        target_kl_value = self.f_target_kl(self.n_update)
        verbose = self.debug and (pid is None or pid == 0)
        if verbose:
            logging.debug("\nbatch_size: {}\nkl_target: {}\n".format(num_samples, target_kl_value))
            surr_o, kl_o, ent_o = run_batched([self.policy.trad_surr_loss, self.policy.kl, self.policy.ent], feed,
                                              num_samples,
                                              self.session,
                                              minibatch_size=self.minibatch_size,
                                              extra_input={
                                                  self.target_kl_sym: target_kl_value})
            logging.debug("\nold ppo_surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))

        # grads = run_batched(self.k_grads, feed, num_samples, self.session,
        #                     minibatch_size=self.minibatch_size,
        #                     extra_input={self.target_kl_sym: target_kl_value},
        #                     average=False
        #                     )
        # grad_dict = {p: v for (p, v) in zip(self.k_grads_ph, grads)}
        # _ = self.session.run(self.k_apply_grad, feed_dict={**feed, **grad_dict})

        _ = self.session.run(self.k_update_op, feed_dict=feed)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl_new = run_batched(self.policy.kl, feed, num_samples, self.session,
                             minibatch_size=self.minibatch_size,
                             extra_input={
                                 self.target_kl_sym: target_kl_value})

        if kl_new > target_kl_value * 2:
            if verbose: logging.debug("kl too high")
            self.session.run(tf.assign(self.k_stepsize, tf.maximum(min_stepsize, self.k_stepsize / 1.5)))
        elif kl_new < target_kl_value / 2:
            if verbose: logging.debug("kl too low")
            self.session.run(tf.assign(self.k_stepsize, tf.minimum(max_stepsize, self.k_stepsize * 1.5)))
        else:
            if verbose: logging.debug("kl just right!")
        if verbose:
            surr_new, ent_new = run_batched([self.policy.trad_surr_loss, self.policy.ent], feed, num_samples,
                                            self.session,
                                            minibatch_size=self.minibatch_size,
                                            extra_input={
                                                self.target_kl_sym: target_kl_value})
            logging.debug("\nnew ppo_surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))

    def fit_pg(self, feed, num_samples, pid=None):

        target_kl_value = self.f_target_kl(self.n_update) if self.f_target_kl is not None else None
        if self.debug:
            logging.debug("\nbatch_size: {}".format(num_samples))
            surr_o, kl_o, ent_o = run_batched([self.policy.trad_surr_loss, self.policy.kl, self.policy.ent], feed,
                                              num_samples,
                                              self.session,
                                              minibatch_size=self.minibatch_size,
                                              extra_input={})
            logging.debug("\nold_surr: {}\told kl: {}\told ent: {}".format(surr_o, kl_o, ent_o))
        _ = tf_run_batched(self.p_accum_op, self.p_accum_reset, feed, num_samples, self.session,
                           minibatch_size=self.minibatch_size,
                           extra_input={self.target_kl_sym: target_kl_value,
                                        self.p_beta: self.p_beta_value})
        g_norm, _ = self.session.run([self.g_norm, self.p_apply_grad],
                                     feed_dict={self.p_sym_step_size: self.p_step_size,
                                                self.p_beta: self.p_beta_value, })

        surr_new, kl_new, ent_new = run_batched([self.policy.trad_surr_loss, self.policy.kl, self.policy.ent], feed,
                                                num_samples,
                                                self.session,
                                                minibatch_size=self.minibatch_size, )
        if self.debug:
            logging.debug("\nnew_surr: {}\tnew_kl: {}\tnew_ent: {}".format(surr_new, kl_new, ent_new))
            logging.debug("\ngnorm: {}".format(g_norm))
        if self.f_target_kl is not None:
            if kl_new > target_kl_value * 2:
                self.p_beta_value = np.minimum(self.p_beta_max,
                                               1.5 / 2 * (kl_new / target_kl_value) * self.p_beta_value)
                if self.p_beta_value > self.p_beta_max / 1.2:
                    self.p_step_size = np.maximum(self.p_min_step_size, self.p_step_size / 1.5)
                logging.debug('beta -> %s' % self.p_beta_value)
                logging.debug('step_size -> %s' % self.p_step_size)
            elif kl_new < target_kl_value / 2:
                self.p_beta_value = np.maximum(self.p_beta_min, self.p_beta_value / 1.5)
                if self.p_beta_value < 1.2 * self.p_beta_min:
                    self.p_step_size = np.minimum(self.p_max_step_size, 1.5 * self.p_step_size)
                logging.debug('beta -> %s' % self.p_beta_value)
                logging.debug('step_size -> %s' % self.p_step_size)
            else:
                logging.debug('beta OK = %s' % self.p_beta_value)
                logging.debug('step_size OK = %s' % self.p_step_size)

    def train(self, paths, pid=None):
        if self.const_action is not None:
            return
        if paths is not None and self.is_decider:
            logging.debug("root at t\t{}".format(self.n_update))
            norm_logits = paths["logits"] - np.logaddexp.reduce(paths["logits"], axis=1)[:, np.newaxis]
            logging.info("pi:{}".format(np.mean(np.exp(norm_logits), axis=0)))
        if self.is_switcher_with_init_len:
            logging.info("ave subt:{}".format(np.mean(paths["observation"][-1][:, 1])))
        if paths is not None:
            feed, feed_critic, extra_data = self.concat_paths(paths)
            mean_t_reward = extra_data["rewards"].mean()
            logging.info("name:\t{0} mean_r_t:\t{1:.4f}".format(self.name, mean_t_reward))
        else:
            feed, mean_t_reward, feed_critic = None, None, None

        if self.should_train and self.f_train_this_epoch(self.n_update):
            if paths is None:
                logging.debug("No training data for {}".format(self.name))
            else:
                logging.debug("-------------------------------------------")
                logging.debug("training model:\t{} at t\t{}".format(self.name,self.n_update))
                batch_size = feed[self.policy.advant].shape[0]
                self.handle_model_saving(mean_t_reward)

                self.critic_lock.acquire_write()
                if self.should_update_critic:
                    self.critic.fit(feed_critic, update_mode="full", num_pass=1, pid=pid)
                self.critic_lock.release_write()

                # if self.debug:
                #     advant_n = feed[self.policy.advant]
                #     # logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))
                #     # action_dist_logstds_n = feed[self.net.dist_vars["logstd"]]
                #     # logging.debug("state max: {}\n min: {}".format(state_input.max(axis=0), state_input.min(axis=0)))
                #     if hasattr(self.act_space, "low"):
                #         logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
                #         # logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
                if self.should_update_policy and batch_size > 0:
                    self.policy_lock.acquire_write()
                    if self.should_update_policy:
                        self.fit_policy(feed, batch_size, pid=pid)

                    self.policy_lock.release_write()
                    self.n_fit += 1
        self.increment_n_update()
