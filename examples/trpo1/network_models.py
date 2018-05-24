from __future__ import absolute_import
import tensorflow as tf
from scaling_orth import ScalingOrth
import numpy as np

from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st, logit
import logging
from prob_types import DiagonalGaussian, Categorical, logits_from_ter_categorical
dtype = tf.float32


# from cnn import cnn_network
# from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
# import baselines.common.tf_util as U
# from tf_utils import aggregate_feature, lrelu
# import prettytensor as pt


class MultiNetwork(object):
    def __init__(self, scope, observation_space, action_space,
                 n_imgfeat=1, extra_feaatures=[], st_enabled=None, img_enabled=None,
                 comb_method=aggregate_feature,
                 initializer=ScalingOrth,
                 activation=tf.tanh,
                 cnn_trainable=True,
                 distribution=DiagonalGaussian(1),
                 session=None,
                 f_build_cnn=None,
                 is_switcher_with_init_len=0,
                 use_wasserstein=False
                 ):
        self.comb_method = comb_method
        self.is_switcher_with_init_len = is_switcher_with_init_len
        self.initializer = initializer
        self.session = session
        self.use_wasserstein = use_wasserstein
        local_scope = scope
        with tf.variable_scope(local_scope) as this_scope:
            self.state_input = tf.placeholder(
                dtype, shape=(None,) + observation_space[0].shape, name="%s_state" % scope)
            self.checked_state_input = tf.check_numerics(self.state_input, "state is nan")
            if n_imgfeat != 0:
                self.img_input = \
                    tf.placeholder(tf.uint8, shape=(None,) + observation_space[1].shape, name="%s_img" % scope)
                self.img_float = tf.cast(self.img_input, tf.float32) / 255
            # else:
            #     self.img_input = 0
            if self.is_switcher_with_init_len:
                self.hrl_meta_input = tf.placeholder(tf.float32,
                                                     shape=(None, *observation_space[-1].shape),
                                                     name="hrl_meta_input")
            if hasattr(action_space, 'low'):
                self.action_n = tf.placeholder(dtype, shape=(None,) + action_space.shape, name="%s_action" % scope)
            else:
                self.action_n = tf.placeholder(tf.int32, shape=(None,), name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)
            self.advant = tf.check_numerics(self.advant, "advant nan")

            # self.st_enabled = st_enabled
            # self.img_enabled = img_enabled
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + observation_space[0].shape, name='st_enabled')
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')

            if len(extra_feaatures) > 0:
                self.full_feature = self.comb_method(self.st_enabled * self.checked_state_input, extra_feaatures[0])
            else:
                if n_imgfeat != 0:
                    assert f_build_cnn is not None
                    img_net_layers, cnn_weights, img_fc_weights = f_build_cnn(self.img_float)
                    self.cnn_weights = cnn_weights
                    self.img_fc_weights = img_fc_weights
                    self.img_var_list = cnn_weights + img_fc_weights

                    if n_imgfeat < 0:
                        self.image_features = tf.layers.flatten(img_net_layers[-1])
                    else:
                        self.image_features = img_net_layers[-1]
                    self.full_feature = self.comb_method(self.st_enabled * self.checked_state_input,
                                                         self.img_enabled[:, tf.newaxis] * self.image_features)
                    if self.is_switcher_with_init_len:
                        self.full_feature = tf.concat(axis=1,
                                                      values=[self.full_feature, self.hrl_meta_input])

                else:
                    self.full_feature = self.checked_state_input
                    self.cnn_weights = []
                    self.img_fc_weights = []

            hidden_sizes = (64, 64)
            # logging.info("policy hidden sizes: {}".format(hidden_sizes))
            self.fc_layers = [self.full_feature]
            if isinstance(distribution, DiagonalGaussian):
                fc_activation = tf.tanh
            else:
                fc_activation = activation
                logging.info("policy activation: {}".format(fc_activation))
            for hid in hidden_sizes:
                current_fc = tf.layers.dense(self.fc_layers[-1], hid, activation=fc_activation,
                                             kernel_initializer=
                                             self.initializer()
                                             )
                self.fc_layers.append(current_fc)
            batch_size = tf.shape(self.state_input)[0]
            self.batch_size_float = tf.cast(batch_size, tf.float32)

            self.distribution = distribution
            if not self.is_switcher_with_init_len:
                self.dist_vars, self.old_vars, self.sampled_action, self.interm_vars = \
                    self.distribution.create_dist_vars(self.fc_layers[-1])
            else:
                root_logits = tf.layers.dense(self.fc_layers[-1], 1,
                                              kernel_initializer=self.initializer())
                max_length = self.is_switcher_with_init_len - 0.5
                with tf.variable_scope("switch_model") as time_scope:
                    self.time_weight = tf.get_variable(name="time_weight", initializer=tf.constant(10.0),
                                                       trainable=False)
                    self.time_offset = tf.get_variable(name="time_offset", initializer=tf.constant(max_length),
                                                  trainable=False)
                    time_logit = self.time_weight * (self.hrl_meta_input[:, 1:2] - self.time_offset)
                    self.fixed_prob_ter_logit = tf.get_variable("fix_ter_prob",
                                                                initializer=tf.constant(logit(0.01),
                                                                                        dtype=tf.float32),
                                                                dtype=tf.float32)

                    self.fixed_ter_weight = tf.get_variable("fix_ter_w", initializer=tf.constant(0.0, dtype=tf.float32),
                                                            dtype=tf.float32)
                    final_terlogit = self.fixed_ter_weight * self.fixed_prob_ter_logit + \
                                     (1 - self.fixed_ter_weight) * time_logit

                self.full_logits = logits_from_ter_categorical(root_logits, final_terlogit,
                                                               is_initial_step=self.hrl_meta_input[:, 2:3])
                assert isinstance(self.distribution, Categorical)
                self.dist_vars, self.old_vars, self.sampled_action, self.interm_vars = \
                    self.distribution.create_dist_vars(logits=self.full_logits)
            self.mean_loglike = - tf.reduce_mean(
                self.distribution.kf_loglike(self.action_n, self.dist_vars, self.interm_vars))

            self.new_likelihood_sym = tf.check_numerics(
                self.distribution.log_likelihood_sym(self.action_n, self.dist_vars),
                "new logpi nan")
            self.old_likelihood = tf.check_numerics(self.distribution.log_likelihood_sym(self.action_n, self.old_vars),
                                                    "old logpi nan")
            # self.old_likelihood = tf.check_numerics(
            #     tf.maximum(self.distribution.log_likelihood_sym(self.action_n, self.old_vars), np.log(1e-8)),
            #                                         "old logpi nan")
            eps = 1e-8
            self.ratio_n = tf.check_numerics(tf.exp(self.new_likelihood_sym - self.old_likelihood), "ratio is nan")
            self.var_list = tf.trainable_variables(this_scope.name)
            if not cnn_trainable:
                self.var_list = [v for v in self.var_list if not (v in self.cnn_weights or v in self.img_fc_weights)]
            self.p_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list])
            self.PPO_eps = 0.2
            self.clipped_ratio = tf.clip_by_value(self.ratio_n, 1.0 - self.PPO_eps, 1.0 + self.PPO_eps)
            self.surr_n = self.ratio_n * self.advant
            self.clipped_surr = self.clipped_ratio * self.advant
            self.ppo_surr = tf.check_numerics(-tf.reduce_mean(tf.minimum(self.surr_n, self.clipped_surr)),
                                              "ppo loss is nan")  # Surrogate loss

            # Sampled loss of the policy

            self.trad_loss = - tf.reduce_mean(self.new_likelihood_sym * self.advant)
            log_ppo_max_ratio = np.log(1.0 + self.PPO_eps)
            log_ppo_min_ratio = np.log(1.0 - self.PPO_eps)
            clipped_new_likelihoold = tf.clip_by_value(self.new_likelihood_sym,
                                                       self.old_likelihood + log_ppo_min_ratio,
                                                       self.old_likelihood + log_ppo_max_ratio)
            self.ppo_trad_loss = tf.check_numerics(
                -tf.reduce_mean(tf.minimum(self.new_likelihood_sym * self.advant,
                                           clipped_new_likelihoold * self.advant)),
                "ppo_trad nan")
            if self.use_wasserstein:
                self.wassersteins = self.distribution.wasserstein_sym(self.old_vars, self.dist_vars)
                self.kl = tf.reduce_mean(self.wassersteins)
                self.loss_sampled = self.kl
            else:
                self.kls = self.distribution.kl_sym(self.old_vars, self.dist_vars)
                self.kl = tf.reduce_mean(self.kls)
                self.loss_sampled = self.mean_loglike

            ent_n = self.distribution.entropy(self.dist_vars)  # - self.new_likelihood_sym
            self.ent = tf.reduce_sum(ent_n) / self.batch_size_float
            self.rl_losses = {"PPO": self.ppo_surr, "TRAD": self.trad_loss, "PPO_TRAD": self.ppo_trad_loss}
            self.losses = [self.ppo_surr, self.kl, self.ent]
            self.reset_exp = self.distribution.reset_exp(self.interm_vars)
            if self.is_switcher_with_init_len:
                self.switch_prob = tf.sigmoid(self.full_logits[:, 1:2])
                self.switcher_cost = tf.reduce_mean(self.switch_prob)
