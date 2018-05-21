from __future__ import absolute_import
from arena.models.model import ModelWithCritic
import tensorflow as tf
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st, explained_variance_batched, tf_run_batched, batch_run_forward, MiniBatchAccumulator
# from baseline import Baseline
from multi_baseline import MultiBaseline
from prob_types import DiagonalGaussian, Categorical
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
from concurrent.futures import ThreadPoolExecutor
import os
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
                 n_imgfeat=0,
                 f_target_kl=None,
                 lr=0.0001,
                 critic_lr=0.0003,
                 npass=1,
                 minibatch_size=64,
                 multi_update=False,
                 mode="ACKTR",
                 loss_type="PPO",
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
                 max_grad_norm=0.5,
                 is_switcher_with_init_len=0,
                 switcher_cost_k=0.01,
                 is_decider=False,
                 reset_exp=False,
                 const_action=None
                 ):
        ModelWithCritic.__init__(self, observation_space, action_space)
        self.ob_space = observation_space
        self.act_space = action_space
        self.multi_update = multi_update
        args = locals()
        logging.debug("model args:\n {}".format(args))

        # store constants
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

        def initializer():
            # return ScalingOrth(scale=1.0)
            return tf.variance_scaling_initializer(scale=1.0, mode="fan_avg")

        def f_build_img_net(t_input):
            return cnn_network(t_input, conv_sizes, cnn_activation=tf.nn.leaky_relu,
                               fc_sizes=cnn_fc_feat, fc_activation=tf.nn.leaky_relu,
                               initializer_fn=initializer)

        self.name = name if not self.is_switcher_with_init_len else name + "_switcher"
        self.is_decider = is_decider
        self.npass = npass
        self.critic = MultiBaseline(session=self.session, observation_space=self.ob_space,
                                    minibatch_size=minibatch_size,
                                    main_scope=self.name + "_critic",
                                    activation=tf.nn.leaky_relu,
                                    n_imgfeat=self.n_imgfeat,
                                    comb_method=self.comb_method,
                                    cnn_trainable=cnn_trainable,
                                    f_build_cnn=f_build_img_net,
                                    is_switcher_with_init_len=is_switcher_with_init_len,
                                    initializer=initializer,
                                    lr=critic_lr)
        if hasattr(self.act_space, "low"):
            self.distribution = DiagonalGaussian(dim=self.act_space.low.shape[0])
        else:
            self.distribution = Categorical(num_cat=self.act_space.n)

        self.theta = None
        WASS_POSTFIX = "_WASS"

        use_wasserstein = (self.mode == "PG" and loss_type.endswith(WASS_POSTFIX))

        self.policy = MultiNetwork(scope=self.name + "_policy",
                                   observation_space=self.ob_space,
                                   action_space=self.act_space,
                                   n_imgfeat=self.n_imgfeat,
                                   extra_feaatures=[],
                                   comb_method=self.comb_method,
                                   distibution=self.distribution,
                                   session=self.session,
                                   cnn_trainable=cnn_trainable,
                                   f_build_cnn=f_build_img_net,
                                   initializer=initializer,
                                   is_switcher_with_init_len=is_switcher_with_init_len,
                                   use_wasserstein=use_wasserstein
                                   )
        self.executer_net = self.policy

        self.losses = self.policy.losses
        var_list = self.policy.var_list

        self.target_kl_initial = None if f_target_kl is None else f_target_kl(0)
        self.target_kl_sym = tf.placeholder(shape=[], dtype=tf.float32, name="target_kl")
        self.ent_k = ent_k
        self.ent_loss = 0 if self.ent_k == 0 else -self.ent_k * self.policy.ent
        self.fit_policy = None

        self.loss_type = loss_type[:-len(WASS_POSTFIX)] if loss_type.endswith(WASS_POSTFIX) else loss_type

        self.rl_loss = self.policy.rl_losses[self.loss_type]
        if self.is_switcher_with_init_len:
            self.switcher_cost_k = switcher_cost_k
            self.rl_loss = self.rl_loss + self.switcher_cost_k + self.policy.switcher_cost
        if should_train:
            if self.mode == "ACKTR":
                self.k_stepsize = tf.Variable(initial_value=np.float32(lr), name='stepsize')
                self.k_momentum = 0.9
                self.k_optim = kfac.KfacOptimizer(learning_rate=self.k_stepsize,
                                                  cold_lr= self.k_stepsize * (1 - self.k_momentum),
                                                  momentum=self.k_momentum,
                                                  kfac_update=2,
                                                  epsilon=1e-2, stats_decay=0.99,
                                                  async=True, cold_iter=1,
                                                  weight_decay_dict={}, max_grad_norm=None)

                self.k_final_loss = self.rl_loss + self.ent_loss

                self.k_update_op, self.k_q_runner = self.k_optim.minimize(self.k_final_loss,
                                                                          self.policy.mean_loglike,
                                                                          var_list=self.policy.var_list)
                self.k_enqueue_threads = []
                self.k_coord = tf.train.Coordinator()
                self.fit_policy = self.fit_acktr
            elif self.mode == "PG":
                with tf.variable_scope("pg") as scope:
                    self.pg_npass = 10
                    self.pg_minibatch_size = minibatch_size
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

                    self.p_grads_accumulator = MiniBatchAccumulator(self.p_grads)
                    self.p_accum_op = self.p_grads_accumulator.gen_accum_op(self.p_grads)

                    if max_grad_norm is not None:
                        p_final_grads, self.g_norm = tf.clip_by_global_norm(self.p_grads_accumulator.accum_var,
                                                                            max_grad_norm)
                    else:
                        p_final_grads = self.p_grads_accumulator.accum_var
                    p_grads_var = [(g, v) for g, v in zip(p_final_grads, self.policy.var_list)]
                    p_apply_grad = self.pg_optim.apply_gradients(p_grads_var)
                    self.p_apply_reset_accum = \
                        self.p_grads_accumulator.gen_apply_reset_op(p_apply_grad)

                self.fit_policy = self.fit_pg
            else:
                raise NotImplementedError

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
        self.critic_update_executor = ThreadPoolExecutor(max_workers=1)

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
        if os.path.exists(self.model_load_path + ".index"):
            try:
                self.full_model_saver.restore(self.session, self.model_load_path)
            except (ValueError):
                logging.debug("failed to load {}".format(self.model_load_path))
        else:
            logging.warning("path doesn't exist {}".format(self.model_load_path))
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

        extra = {"return": returns, "reward": rewards, "time": times}

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
            if mean_t_reward >= 0.9 * self.best_mean_reward:
                self.best_mean_reward = mean_t_reward
                logging.debug("Saving {} with r: {}".format(self.model_save_path, self.best_mean_reward))
                self.full_model_saver.save(self.session, self.model_save_path, write_state=False)
                self.last_save = self.n_update

    def print_stat(self, feed, num_samples, tag="old"):

        surr, kl, ent = run_batched([self.policy.trad_loss, self.policy.kl, self.policy.ent], feed,
                                    num_samples,
                                    self.session,
                                    minibatch_size=self.minibatch_size,
                                    extra_input={})
        if self.debug:
            logging.debug("\n{0}_surr: {1}\t{0}_kl: {2}\t{0}_ent: {3}".format(tag, surr, kl, ent))
        return surr, kl, ent

    def fit_acktr(self, feed, num_samples, pid=None):
        target_kl_value = self.f_target_kl(self.n_update)
        verbose = self.debug and (pid is None or pid == 0)

        _, _, _ = self.print_stat(feed, num_samples, tag="old")

        _ = self.session.run(self.k_update_op, feed_dict=feed)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)

        _, kl_new, _ = self.print_stat(feed, num_samples, tag="new")

        if kl_new > target_kl_value * 2:
            if verbose: logging.debug("kl too high")
            self.session.run(tf.assign(self.k_stepsize, tf.maximum(min_stepsize, self.k_stepsize / 1.5)))
        elif kl_new < target_kl_value / 2:
            if verbose: logging.debug("kl too low")
            self.session.run(tf.assign(self.k_stepsize, tf.minimum(max_stepsize, self.k_stepsize * 1.5)))
        else:
            if verbose: logging.debug("kl just right!")

    def fit_pg(self, feed, num_samples, pid=None):

        target_kl_value = self.f_target_kl(self.n_update) if self.f_target_kl is not None else None
        _, _, _ = self.print_stat(feed, num_samples, tag="old")
        for pass_i in range(self.pg_npass):
            if self.multi_update:
                ids = np.random.permutation(np.arange(num_samples))
                for start in range(0, num_samples, self.minibatch_size):
                    end = start + self.minibatch_size if num_samples - start < 2 * self.minibatch_size else num_samples
                    slc = ids[start:end]
                    this_feed = {k: v[slc] for k, v in feed.items()}
                    _ = self.session.run(self.p_accum_op,
                                         feed_dict={**this_feed,
                                                    self.p_sym_step_size: self.p_step_size,
                                                    self.p_beta: self.p_beta_value, })
                    g_norm, _ = self.session.run([self.g_norm, self.p_apply_reset_accum],
                                                 feed_dict={self.p_sym_step_size: self.p_step_size,
                                                            self.p_beta: self.p_beta_value, })
            else:
                _ = tf_run_batched(self.p_accum_op, feed, num_samples, self.session,
                                   minibatch_size=self.minibatch_size,
                                   extra_input={self.target_kl_sym: target_kl_value,
                                                self.p_beta: self.p_beta_value})
                g_norm, _ = self.session.run([self.g_norm, self.p_apply_reset_accum],
                                             feed_dict={self.p_sym_step_size: self.p_step_size,
                                                        self.p_beta: self.p_beta_value, })

        _, kl_new, _ = self.print_stat(feed, num_samples, tag="new")
        if self.debug:
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

    def fit_critic(self, feed_critic, pid=None):
        self.critic_lock.acquire_write()
        self.critic.fit(feed_critic, update_mode="full", num_pass=self.npass, pid=pid)
        self.critic_lock.release_write()

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
            mean_t_reward = extra_data["return"].mean()
            logging.info("name: {0} mean_r_t: {1:.4f}".format(self.name, mean_t_reward))
        else:
            feed, mean_t_reward, feed_critic = None, None, None

        if self.should_train and self.f_train_this_epoch(self.n_update):
            if paths is None:
                logging.debug("No training data for {}".format(self.name))
            else:
                logging.debug("-------------------------------------------")
                logging.debug("training model: {} at t\t{}".format(self.name, self.n_update))
                batch_size = feed[self.policy.advant].shape[0]
                self.handle_model_saving(mean_t_reward)

                if self.should_update_policy and batch_size > 0:
                    self.policy_lock.acquire_write()
                    if self.should_update_policy:
                        if self.debug:
                            logging.debug("\nbatch_size: {}".format(batch_size))
                            if hasattr(self.act_space, "low"):
                                logging.debug("std: {}".format(np.mean(np.exp(np.ravel(paths["logstd"])))))
                        self.fit_policy(feed, batch_size, pid=pid)

                    self.policy_lock.release_write()
                    self.n_fit += 1

                if self.should_update_critic:
                    self.critic_update_executor.submit(self.fit_critic, feed_critic, pid)
        self.increment_n_update()
