from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st
import logging
from prob_types import DiagonalGaussian
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
                 f_build_cnn=None):
        self.comb_method = comb_method
        self.min_std = min_std

        self.session = session
        local_scope = scope
        with tf.variable_scope(local_scope) as this_scope:
            self.state_input = tf.placeholder(
                dtype, shape=(None,) + observation_space[0].shape, name="%s_state" % scope)
            if n_imgfeat != 0:
                self.img_input = \
                    tf.placeholder(tf.uint8, shape=(None,) + observation_space[1].shape, name="%s_img" % scope)
                self.img_float = tf.cast(self.img_input, tf.float32) / 255
            # else:
            #     self.img_input = 0
            if hasattr(action_space, 'low'):
                self.action_n = tf.placeholder(dtype, shape=(None,) + action_space.shape, name="%s_action" % scope)
            else:
                self.action_n = tf.placeholder(tf.int32, shape=(None,), name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)

            # self.st_enabled = st_enabled
            # self.img_enabled = img_enabled
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + observation_space[0].shape, name='st_enabled')
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')

            if len(extra_feaatures) > 0:
                self.full_feature = self.comb_method(self.st_enabled * self.state_input, extra_feaatures[0])
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
                    self.full_feature = self.comb_method(self.st_enabled * self.state_input,
                                                         self.img_enabled[:, tf.newaxis] * self.image_features)

                else:
                    self.full_feature = self.state_input
                    self.cnn_weights = []
                    self.img_fc_weights = []

            hidden_sizes = (64, 64)
            logging.info("policy hidden sizes: {}".format(hidden_sizes))
            self.fc_layers = [self.full_feature]
            for hid in hidden_sizes:
                current_fc = tf.layers.dense(self.fc_layers[-1], hid, activation=tf.tanh,
                                             kernel_initializer=
                                             tf.variance_scaling_initializer(
                                                 scale=1.0, mode="fan_avg", distribution="normal", dtype=dtype)
                                             )
                self.fc_layers.append(current_fc)
            batch_size = tf.shape(self.state_input)[0]
            self.batch_size_float = tf.cast(batch_size, tf.float32)

            self.distribution = distibution

            self.dist_vars, self.old_vars, self.sampled_action, self.interm_vars = \
                self.distribution.create_dist_vars(self.fc_layers[-1])
            self.mean_loglike = - tf.reduce_mean(
                self.distribution.kf_loglike(self.action_n, self.dist_vars, self.interm_vars))

            self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.action_n, self.dist_vars)
            self.old_likelihood = self.distribution.log_likelihood_sym(self.action_n, self.old_vars)

            self.ratio_n = tf.exp(self.new_likelihood_sym - self.old_likelihood)
            self.var_list = tf.trainable_variables(this_scope.name)
            if not cnn_trainable:
                self.var_list = [v for v in self.var_list if not (v in self.cnn_weights or v in self.img_fc_weights)]
            self.p_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list])
            self.PPO_eps = 0.2
            self.clipped_ratio = tf.clip_by_value(self.ratio_n, 1.0 - self.PPO_eps, 1.0 + self.PPO_eps)
            self.surr_n = self.ratio_n * self.advant
            self.clipped_surr = self.clipped_ratio * self.advant
            self.ppo_surr = -tf.reduce_mean(tf.minimum(self.surr_n, self.clipped_surr))  # Surrogate loss

            # Sampled loss of the policy

            self.trad_surr_loss = - tf.reduce_mean(self.new_likelihood_sym * self.advant)

            self.kl = tf.reduce_mean(self.distribution.kl_sym(self.old_vars, self.dist_vars))
            ent_n = self.distribution.entropy(self.dist_vars)  # - self.new_likelihood_sym
            self.ent = tf.reduce_sum(ent_n) / self.batch_size_float
            self.losses = [self.ppo_surr, self.kl, self.ent]




            # def get_action_dist_means_n(self, session, obs):
            #     return session.run(self.mean_n,
            #                        {self.state_input: obs[0],
            #                         self.img_input: obs[1]})
