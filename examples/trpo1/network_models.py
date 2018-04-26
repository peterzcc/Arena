from __future__ import absolute_import
import tensorflow as tf
from scaling_orth import ScalingOrth
import numpy as np

from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st, logit
import logging
from prob_types import DiagonalGaussian, CategoricalWithProb, Categorical, logits_from_cond_categorical
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
                 cnn_trainable=True,
                 min_std=1e-6,
                 distibution=DiagonalGaussian(1),
                 session=None,
                 f_build_cnn=None,
                 is_flexible_hrl_model=False
                 ):
        self.comb_method = comb_method
        self.min_std = min_std
        self.is_flexible_hrl_model = is_flexible_hrl_model

        self.session = session
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
            if self.is_flexible_hrl_model:
                self.hrl_meta_input = tf.placeholder(tf.float32,
                                                     shape=(None, *observation_space[2].shape),
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

                    if n_imgfeat < 0:
                        self.image_features = tf.layers.flatten(img_net_layers[-1])
                    else:
                        self.image_features = img_net_layers[-1]
                    self.full_feature = self.comb_method(self.st_enabled * self.checked_state_input,
                                                         self.img_enabled[:, tf.newaxis] * self.image_features)
                    if self.is_flexible_hrl_model:
                        self.full_feature = tf.concat(axis=1,
                                                      values=[self.full_feature, self.hrl_meta_input])

                else:
                    self.full_feature = self.checked_state_input
                    self.cnn_weights = []
                    self.img_fc_weights = []

            hidden_sizes = (64, 64)
            logging.info("policy hidden sizes: {}".format(hidden_sizes))
            self.fc_layers = [self.full_feature]
            for hid in hidden_sizes:
                current_fc = tf.layers.dense(self.fc_layers[-1], hid, activation=tf.tanh,
                                             kernel_initializer=
                                             ScalingOrth(
                                                 scale=1.0, dtype=dtype)
                                             )
                self.fc_layers.append(current_fc)
            batch_size = tf.shape(self.state_input)[0]
            self.batch_size_float = tf.cast(batch_size, tf.float32)

            self.distribution = distibution
            if not self.is_flexible_hrl_model:
                self.dist_vars, self.old_vars, self.sampled_action, self.interm_vars = \
                    self.distribution.create_dist_vars(self.fc_layers[-1])
            else:
                root_logits = tf.layers.dense(self.fc_layers[-1], action_space.n - 1,
                                              kernel_initializer=ScalingOrth(scale=1.0, dtype=dtype))
                max_length = 50
                with tf.variable_scope("ter_model") as time_scope:
                    self.time_weight = tf.get_variable(name="time_weight", initializer=tf.constant(10.0),
                                                       trainable=False)
                    self.time_offset = tf.get_variable(name="time_offset", initializer=tf.constant(max_length),
                                                  trainable=False)
                    time_logit = -self.time_weight * (self.hrl_meta_input[:, 1:2] - self.time_offset)
                    self.fixed_prob_ter_logit = tf.get_variable("fix_ter_prob",
                                                                initializer=tf.constant(logit(0.01),
                                                                                        dtype=tf.float32),
                                                                dtype=tf.float32)
                    cont_prob_offset = 1.0 - self.fixed_prob_ter_logit
                    self.fixed_ter_weight = tf.get_variable("fix_ter_w", initializer=tf.constant(1.0, dtype=tf.float32),
                                                            dtype=tf.float32)
                    final_contlogit = self.fixed_ter_weight * cont_prob_offset + \
                                  (1-self.fixed_ter_weight)*time_logit

                full_logits = logits_from_cond_categorical(root_logits, final_contlogit,
                                                           is_initial_step=self.hrl_meta_input[:, 2:3])
                assert isinstance(self.distribution, Categorical)
                self.dist_vars, self.old_vars, self.sampled_action, self.interm_vars = \
                    self.distribution.create_dist_vars(logits=full_logits)
            self.mean_loglike = - tf.reduce_mean(
                self.distribution.kf_loglike(self.action_n, self.dist_vars, self.interm_vars))

            self.new_likelihood_sym = tf.check_numerics(
                self.distribution.log_likelihood_sym(self.action_n, self.dist_vars),
                "new logpi nan")
            self.old_likelihood = tf.check_numerics(
                tf.maximum(self.distribution.log_likelihood_sym(self.action_n, self.old_vars), np.log(1e-8)),
                                                    "old logpi nan")

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

            self.trad_surr_loss = - tf.reduce_mean(self.new_likelihood_sym * self.advant)
            self.kls = self.distribution.kl_sym(self.old_vars, self.dist_vars)
            self.kl = tf.reduce_mean(self.kls)

            ent_n = self.distribution.entropy(self.dist_vars)  # - self.new_likelihood_sym
            self.ent = tf.reduce_sum(ent_n) / self.batch_size_float
            self.losses = [self.ppo_surr, self.kl, self.ent]
            self.reset_exp = self.distribution.reset_exp(self.interm_vars)
