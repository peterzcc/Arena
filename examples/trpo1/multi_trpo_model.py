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
from cnn import ConvAutoencorder
from baselines import common
from baselines.common import tf_util as U
from baselines.acktr import kfac


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
                 f_target_kl=None,
                 minibatch_size=128,
                 gamma=0.995,
                 gae_lam=0.97,
                 mode="ADA_KL",
                 num_actors=1,
                 f_batch_size=None,
                 batch_mode="episode",
                 recompute_old_dist=False,
                 update_per_epoch=4,
                 kl_history_length=1,
                 ent_k=0,
                 comb_method=aggregate_feature,
                 n_ae_train=0,
                 train_feat=False):
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
        self.min_batch_length = 2500
        self.f_batch_size = f_batch_size
        self.batch_size = self.f_batch_size(0)
        self.f_target_kl = f_target_kl
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
        self.comb_method = comb_method  # aggregate_feature#
        conv_sizes = (((3, 3), 16, 2), ((3, 3), 16, 2), ((3, 3), 4, 2))

        cnn_trainable = True
        self.critic = MultiBaseline(session=self.session, obs_space=self.ob_space,
                                    timestep_limit=timestep_limit,
                                    activation=tf.tanh,
                                    n_imgfeat=self.n_imgfeat,
                                    conv_sizes=conv_sizes,
                                    comb_method=self.comb_method,
                                    cnn_trainable=cnn_trainable)

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
                                    conv_sizes=conv_sizes,  # (((3, 3), 2, 2),),  #
                                    comb_method=self.comb_method,
                                    min_std=min_std,
                                    distibution=self.distribution,
                                    session=self.session,
                                    cnn_trainable=cnn_trainable
                                    )
            self.executer_net = self.net



        self.losses = self.net.losses
        var_list = self.net.var_list

        self.target_kl_initial = f_target_kl(0)
        self.target_kl_sym = tf.placeholder(shape=[], dtype=tf.float32, name="a_target_kl")

        # adaptive kl
        self.a_beta = tf.placeholder(shape=[], dtype=tf.float32, name="a_beta")
        self.a_beta_value = 3
        self.a_eta_value = 50
        self.ent_k = ent_k
        self.a_rl_loss = -tf.reduce_mean(self.net.raw_surr)
        self.ent_loss = 0 if self.ent_k == 0 else -self.ent_k * self.net.ent
        self.a_kl_loss = self.a_beta * self.net.kl + \
                         self.a_eta_value * tf.square(tf.maximum(0.0, self.net.kl - 2.0 * self.target_kl_sym))
        self.a_surr = self.a_rl_loss + self.a_kl_loss + self.ent_loss
        self.a_step_size = 0.0001
        self.a_max_step_size = 0.1
        self.a_min_step_size = 1e-7
        self.a_sym_step_size = tf.placeholder(shape=[], dtype=tf.float32, name="a_step_size")
        self.a_opt = tf.train.AdamOptimizer(learning_rate=self.a_sym_step_size)
        self.a_grad_list = tf.gradients(self.a_surr, var_list)
        self.a_rl_grad = tf.gradients(self.a_rl_loss, var_list)
        self.a_kl_grad = tf.gradients(self.a_kl_loss, var_list)
        self.kl_history_length = kl_history_length
        self.hist_obs0 = []
        self.hist_obs1 = []
        self.hist_st_en = []
        self.hist_img_en = []
        self.a_grad_placeholders = [tf.placeholder(tf.float32, shape=g.shape, name="a_grad_sym")
                                    for g in self.a_grad_list]
        self.a_old_parameters = [tf.Variable(tf.zeros(v.shape, dtype=tf.float32), name="a_old_param") for v in var_list]
        self.a_backup_op = [tf.assign(old_v, v) for (old_v, v) in zip(self.a_old_parameters, var_list)]
        self.a_rollback_op = [[tf.assign(v, old_v) for (old_v, v) in zip(self.a_old_parameters, var_list)]]
        self.a_apply_grad = self.a_opt.apply_gradients(grads_and_vars=zip(self.a_grad_placeholders, var_list))
        self.a_losses = [self.a_rl_loss, self.net.kl, self.net.ent]
        self.a_beta_max = 35.0
        self.a_beta_min = 1.0 / 35.0
        self.update_per_epoch = update_per_epoch

        self.k_stepsize = tf.Variable(initial_value=np.float32(0.03), name='stepsize')
        self.k_momentum = 0.9
        self.k_optim = kfac.KfacOptimizer(learning_rate=self.k_stepsize,
                                          cold_lr=self.k_stepsize * (1 - self.k_momentum), momentum=self.k_momentum,
                                          kfac_update=2,
                                          epsilon=1e-2, stats_decay=0.99,
                                          async=True, cold_iter=1,
                                          weight_decay_dict={}, max_grad_norm=None)
        self.k_final_loss = self.net.trad_surr_loss + self.ent_loss
        self.k_update_op, self.k_q_runner = self.k_optim.minimize(self.k_final_loss,
                                                                  self.net.mean_loglike, var_list=self.net.var_list)
        self.k_enqueue_threads = []
        self.k_coord = tf.train.Coordinator()
        self.train_feat = train_feat
        if self.n_imgfeat != 0:
            if n_imgfeat < 0:
                cnn_fc_feat = (0,)
            else:
                cnn_fc_feat = (64, n_imgfeat,)
            self.autoencoder_net = ConvAutoencorder(input=self.net.img_input,
                                                    conv_sizes=conv_sizes,
                                                    num_fc=cnn_fc_feat)
            self.ae_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.ae_train_op = self.ae_opt.minimize(self.autoencoder_net.reg_loss)
            if self.train_feat:
                self.feat_loss = \
                    tf.reduce_mean(tf.square(self.autoencoder_net.encoder_layers[-1] - self.net.state_input[:, 0:2]))
                self.feat_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.feat_train_op = self.feat_opt.minimize(self.feat_loss, var_list=self.autoencoder_net.fc_weights)
            self.cnn_sync_op = []
            self.cnn_sync_op += [tf.assign(self.net.cnn_weights[i],
                                           self.autoencoder_net.encoder_weights[i]) for i in range(len(conv_sizes))]
            self.cnn_sync_op += [tf.assign(self.critic.cnn_weights[i],
                                           self.autoencoder_net.encoder_weights[i]) for i in range(len(conv_sizes))]
            if self.n_imgfeat > 0:
                self.cnn_sync_op += [tf.assign(self.net.img_fc_weights[i],
                                               self.autoencoder_net.fc_weights[i]) for i in
                                     range(len(self.autoencoder_net.fc_weights))]
                self.cnn_sync_op += [tf.assign(self.critic.img_fc_weights[i],
                                               self.autoencoder_net.fc_weights[i]) for i in
                                     range(len(self.autoencoder_net.fc_weights))]
            self.cnn_saver = tf.train.Saver(var_list=self.autoencoder_net.total_var_list)
        self.n_ae_train = n_ae_train
        self.n_feat_train = self.n_ae_train

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
        self.update_critic = True
        self.update_policy = True
        self.debug = True
        self.recompute_old_dist = recompute_old_dist
        self.session.run(tf.global_variables_initializer())
        for qr in [self.k_q_runner]:
            if (qr != None):
                self.k_enqueue_threads.extend(qr.create_threads(self.session, coord=self.k_coord, start=True))
        if self.mode == "SURP":
            self.restore_from_pretrained_policy = True
            self.update_critic = False
            if self.restore_from_pretrained_policy:
                self.saver.restore(self.session, 'policy_parameter')
                self.session.run(tf.assign(self.target_net.log_vars, self.net.log_vars), feed_dict={})

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

    def predict(self, observation, pid=0):

        if self.num_actors == 1:
            if len(observation[0].shape) == len(self.ob_space[0].shape):
                obs = [np.expand_dims(observation[0], 0)]
                if self.n_imgfeat != 0:
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
                if self.n_imgfeat != 0:
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
            if self.n_imgfeat != 0:
                feed[self.critic.img_input] = obs[1]
                feed[self.executer_net.img_input] = obs[1]
            self.policy_lock.acquire_read()
            action_dist_means_n, action_dist_log_stds_n = \
                self.session.run([self.executer_net.action_dist_means_n, self.executer_net.action_dist_logstds_n],
                                 feed)
            self.act_means = action_dist_means_n
            self.act_logstds = action_dist_log_stds_n
            self.policy_lock.release_read()
        self.execution_barrier.wait()

        # logging.debug("am:{},\nastd:{}".format(action_dist_means_n[0],action_dist_stds_n[0]))

        agent_info = dict(mean=self.act_means[pid, :], log_std=self.act_logstds[pid, :],
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
        if self.n_imgfeat != 0:
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
            if self.n_imgfeat != 0:
                feed_forw[self.critic.img_input] = img_input
                feed_forw[self.net.img_input] = img_input
            action_dist_means_n, action_dist_logstds_n = \
                self.session.run([self.net.action_dist_means_n, self.net.action_dist_logstds_n],
                                 feed_forw)
            # action_dist_logstds_n = np.tile(action_dist_logstds, (state_input.shape[0], 1))
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

        hist_feed = {}
        if self.kl_history_length > 1:
            if len(self.hist_obs0) == self.kl_history_length:
                del self.hist_obs0[0]
                del self.hist_img_en[0]
                del self.hist_st_en[0]
                if self.n_imgfeat != 0:
                    del self.hist_obs1[0]
            self.hist_obs0.append(state_input)
            self.hist_st_en.append(st_enabled)
            self.hist_img_en.append(img_enabled)
            if self.n_imgfeat != 0:
                self.hist_obs1.append(img_input)
            hist_feed = {self.net.state_input: np.concatenate(self.hist_obs0),
                         self.net.st_enabled: np.concatenate(self.hist_st_en),
                         self.net.img_enabled: np.concatenate(self.hist_img_en),
                         }
            if self.n_imgfeat != 0:
                hist_feed[self.net.img_input] = np.concatenate(self.hist_obs1)

            action_dist_means_n, action_dist_logstds = \
                self.session.run([self.net.action_dist_means_n, self.net.action_dist_log_stds_n],
                                 hist_feed)
            action_dist_logstds_n = np.tile(action_dist_logstds, (hist_feed[self.net.st_enabled].shape[0], 1))
            hist_feed.update(
                {
                    self.net.old_dist_means_n: action_dist_means_n,
                    self.net.old_dist_logstds_n: action_dist_logstds_n,
                }
            )

        return feed, path_dict, hist_feed

    def compute_critic(self, states):
        self.critic_lock.acquire_read()
        result = self.critic.predict(states)
        self.critic_lock.release_read()
        return result

    def check_batch_finished(self, time, epis):
        if self.batch_mode == "episode":
            return epis >= self.batch_size and time >= self.min_batch_length
        if self.batch_mode == "timestep":
            if not time <= self.batch_size:
                logging.debug("time: {} \nbatchsize: {}".format(time, self.batch_size))
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
        if self.n_imgfeat != 0:
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

    def increment_n_update(self):
        self.n_update += 1
        self.batch_size = self.f_batch_size(self.n_update)

    def compute_update(self, paths):
        if self.mode == "SURP":
            if self.update_policy:
                self.policy_lock.acquire_write()
                self.supervised_update(paths)
                self.policy_lock.release_write()
            return None, None
        feed, path_dict, hist_feed = self.concat_paths(paths)

        advant_n = feed[self.net.advant]
        state_input = feed[self.net.state_input]
        action_dist_logstds_n = feed[self.net.old_dist_logstds_n]
        batch_size = advant_n.shape[0]
        if self.n_update == 0 and self.n_ae_train == -1:
            self.cnn_saver.restore(self.session, "./cnn_model")
            self.session.run(self.cnn_sync_op)
        if self.n_update < self.n_ae_train + self.n_feat_train:
            ae_feed = {self.net.img_input: feed[self.net.img_input],
                       self.net.state_input: feed[self.net.state_input]}
            training_inds = np.random.permutation(batch_size)
            if self.n_update < self.n_ae_train:
                ae_loss = run_batched(self.autoencoder_net.reg_loss, session=self.session,
                                      feed=ae_feed, N=batch_size,
                                      minibatch_size=self.minibatch_size)
                logging.debug("\nae loss before: {}".format(ae_loss))
                for start in range(0, batch_size, self.minibatch_size):  # TODO: verify this
                    if start > batch_size - 2 * self.minibatch_size:
                        end = batch_size
                    else:
                        end = start + self.minibatch_size
                    slc = training_inds[range(start, end)]
                    this_feed = {k: v[slc] for (k, v) in list(ae_feed.items())}
                    self.session.run(self.ae_train_op, feed_dict=this_feed)
                    if end == batch_size:
                        break
                ae_loss = run_batched(self.autoencoder_net.reg_loss, session=self.session,
                                      feed=ae_feed, N=batch_size,
                                      minibatch_size=self.minibatch_size)
                logging.debug("\nae loss after: {}".format(ae_loss))

            elif self.train_feat:
                ae_loss = run_batched(self.feat_loss, session=self.session,
                                      feed=ae_feed, N=batch_size,
                                      minibatch_size=self.minibatch_size)
                logging.debug("\nfeat loss before: {}".format(ae_loss))
                for start in range(0, batch_size, self.minibatch_size):  # TODO: verify this
                    if start > batch_size - 2 * self.minibatch_size:
                        end = batch_size
                    else:
                        end = start + self.minibatch_size
                    slc = training_inds[range(start, end)]
                    this_feed = {k: v[slc] for (k, v) in list(ae_feed.items())}
                    self.session.run(self.feat_train_op, feed_dict=this_feed)
                    if end == batch_size:
                        break
                ae_loss = run_batched(self.feat_loss, session=self.session,
                                      feed=ae_feed, N=batch_size,
                                      minibatch_size=self.minibatch_size)
                logging.debug("\nfeat loss after: {}".format(ae_loss))

            if self.n_update == self.n_ae_train + self.n_feat_train - 1:
                self.cnn_saver.save(self.session, "./cnn_model")
                self.cnn_saver.restore(self.session, "./cnn_model")
                self.session.run(self.cnn_sync_op)

            self.increment_n_update()
            return None, None
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
            self.critic.fit(path_dict, update_mode="img", num_pass=1)


        if self.update_critic:
            self.critic.fit(path_dict, update_mode="full", num_pass=1)
        self.critic_lock.release_write()

        # logging.debug("advant_n: {}".format(np.linalg.norm(advant_n)))


        if self.debug:
            # logging.debug("state max: {}\n min: {}".format(state_input.max(axis=0), state_input.min(axis=0)))
            logging.debug("act_clips: {}".format(np.sum(concat([path["clips"] for path in paths]))))
            logging.debug("std: {}".format(np.mean(np.exp(np.ravel(action_dist_logstds_n)))))
        if not self.update_policy:
            return None, None

        if self.mode == "ADA_KL":
            self.policy_lock.acquire_write()
            target_kl_value = self.f_target_kl(self.n_update)
            logging.debug("\nbatch_size: {}\nkl_target: {}\n".format(self.batch_size, target_kl_value))
            surr_o, kl_o, ent_o = run_batched(self.a_losses, feed, batch_size, self.session,
                                              minibatch_size=self.minibatch_size,
                                              extra_input={self.a_beta: self.a_beta_value,
                                                           self.target_kl_sym: target_kl_value})
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            # self.session.run(self.a_backup_op)
            early_stopped = False
            kl_new = -1
            for e in range(self.update_per_epoch):
                if self.kl_history_length == 1:
                    grads = run_batched(self.a_grad_list, feed, batch_size, self.session,
                                        minibatch_size=self.minibatch_size,
                                        extra_input={self.a_beta: self.a_beta_value,
                                                     self.target_kl_sym: target_kl_value}
                                        )
                    grad_dict = {p: v for (p, v) in zip(self.a_grad_placeholders, grads)}
                    _ = self.session.run(self.a_apply_grad,
                                         feed_dict={**grad_dict,
                                                    self.a_sym_step_size: self.a_step_size})
                    kl_new = run_batched(self.a_losses[1], feed, batch_size, self.session,
                                         minibatch_size=self.minibatch_size,
                                         extra_input={self.a_beta: self.a_beta_value,
                                                      self.target_kl_sym: target_kl_value})

                else:
                    grads_rl = run_batched(self.a_rl_grad, feed, batch_size, self.session,
                                           minibatch_size=self.minibatch_size,
                                           extra_input={self.a_beta: self.a_beta_value,
                                                        self.target_kl_sym: target_kl_value}
                                           )
                    hist_size = hist_feed[self.net.st_enabled].shape[0]
                    grads_kl = run_batched(self.a_kl_grad, hist_feed, hist_size, self.session,
                                           minibatch_size=self.minibatch_size,
                                           extra_input={self.a_beta: self.a_beta_value,
                                                        self.target_kl_sym: target_kl_value}
                                           )
                    grads = [r + k for (r, k) in zip(grads_rl, grads_kl)]
                    grad_dict = {p: v for (p, v) in zip(self.a_grad_placeholders, grads)}
                    _ = self.session.run(self.a_apply_grad,
                                         feed_dict={**grad_dict,
                                                    self.a_sym_step_size: self.a_step_size})
                    kl_new = run_batched(self.net.kl, hist_feed, hist_size, self.session,
                                         minibatch_size=self.minibatch_size,
                                         extra_input={self.a_beta: self.a_beta_value,
                                                      self.target_kl_sym: target_kl_value})
                if kl_new > target_kl_value * 4:
                    logging.debug("KL too large, early stop")
                    early_stopped = True
                    break

            surr_new, ent_new = run_batched([self.a_losses[0], self.a_losses[2]], feed, batch_size, self.session,
                                            minibatch_size=self.minibatch_size,
                                            extra_input={self.a_beta: self.a_beta_value,
                                                         self.target_kl_sym: target_kl_value})
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))
            if kl_new > target_kl_value * 2:
                self.a_beta_value = np.minimum(self.a_beta_max, 1.5 * self.a_beta_value)
                if self.a_beta_value > self.a_beta_max - 5 or early_stopped:
                    self.a_step_size = np.maximum(self.a_min_step_size, self.a_step_size / 1.5)
                logging.debug('beta -> %s' % self.a_beta_value)
                logging.debug('step_size -> %s' % self.a_step_size)
            elif kl_new < target_kl_value / 2:
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

            self.increment_n_update()
            self.policy_lock.release_write()

            return None, None

        if self.mode == "ACKTR":

            self.policy_lock.acquire_write()
            target_kl_value = self.f_target_kl(self.n_update)
            logging.debug("\nbatch_size: {}\nkl_target: {}\n".format(self.batch_size, target_kl_value))
            surr_o, kl_o, ent_o = run_batched([self.net.trad_surr_loss, self.net.kl, self.net.ent], feed, batch_size,
                                              self.session,
                                              minibatch_size=self.minibatch_size,
                                              extra_input={self.a_beta: self.a_beta_value,
                                                           self.target_kl_sym: target_kl_value})
            logging.debug("\nold surr: {}\nold kl: {}\nold ent: {}".format(surr_o, kl_o, ent_o))
            _ = self.session.run(self.k_update_op, feed_dict=feed)
            min_stepsize = np.float32(1e-8)
            max_stepsize = np.float32(1e0)
            # Adjust stepsize
            kl_new = kl = run_batched(self.a_losses[1], feed, batch_size, self.session,
                                      minibatch_size=self.minibatch_size,
                                      extra_input={self.a_beta: self.a_beta_value,
                                                   self.target_kl_sym: target_kl_value})
            if kl > target_kl_value * 2:
                logging.debug("kl too high")
                self.session.run(tf.assign(self.k_stepsize, tf.maximum(min_stepsize, self.k_stepsize / 1.5)))
            elif kl < target_kl_value / 2:
                logging.debug("kl too low")
                self.session.run(tf.assign(self.k_stepsize, tf.minimum(max_stepsize, self.k_stepsize * 1.5)))
            else:
                logging.debug("kl just right!")

            surr_new, ent_new = run_batched([self.net.trad_surr_loss, self.net.ent], feed, batch_size, self.session,
                                            minibatch_size=self.minibatch_size,
                                            extra_input={self.a_beta: self.a_beta_value,
                                                         self.target_kl_sym: target_kl_value})
            logging.debug("\nnew surr: {}\nnew kl: {}\nnew ent: {}".format(surr_new, kl_new, ent_new))

            self.increment_n_update()
            self.policy_lock.release_write()

            return None, None

    def update(self, diff, new=None):
        pass
        # self.set_params_with_flat_data(new)

        # def __del__(self): TODO: handle destruction
        # self.k_coord.request_stop()
        # self.k_coord.join(self.k_enqueue_threads)
