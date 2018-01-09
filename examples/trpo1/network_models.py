from __future__ import absolute_import
import tensorflow as tf
# from tensorflow.contrib.layers import initializers as tf_init
from tensorflow.contrib.layers import variance_scaling_initializer
# import prettytensor as pt
import numpy as np
from tf_utils import aggregate_feature, lrelu
from tf_utils import GetFlat, SetFromFlat, flatgrad, var_shape, linesearch, cg, run_batched, concat_feature, \
    aggregate_feature, select_st
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
import baselines.common.tf_util as U
import logging
from cnn import cnn_network
dtype = tf.float32
from diagonal_gaussian import DiagonalGaussian



class MultiNetwork(object):
    def __init__(self, scope, observation_space, action_shape,
                 conv_sizes=(((4, 4), 16, 2), ((4, 4), 16, 1)),
                 n_imgfeat=1, extra_feaatures=[], st_enabled=None, img_enabled=None,
                 comb_method=aggregate_feature,
                 cnn_trainable=True,
                 min_std=1e-6,
                 distibution=DiagonalGaussian,
                 session=None):
        self.comb_method = comb_method
        self.min_std = min_std
        self.distribution = distibution
        self.session = session
        local_scope = "%s_shared" % scope
        with tf.variable_scope(local_scope):
            self.state_input = tf.placeholder(
                dtype, shape=(None,) + observation_space[0].shape, name="%s_state" % scope)
            if n_imgfeat != 0:
                self.img_input = \
                    tf.placeholder(tf.float32, shape=(None,) + observation_space[1].shape, name="%s_img" % scope)
            # else:
            #     self.img_input = 0

            self.action_n = tf.placeholder(dtype, shape=(None,) + action_shape, name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)

            self.old_dist_means_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                     name="%s_oldaction_dist_logstds" % scope)
            # self.st_enabled = st_enabled
            # self.img_enabled = img_enabled
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + observation_space[0].shape, name='st_enabled')
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')

            if len(extra_feaatures) > 0:
                self.full_feature = self.comb_method(self.st_enabled * self.state_input, extra_feaatures[0])
            else:
                if n_imgfeat != 0:
                    if n_imgfeat < 0:
                        cnn_fc_feat = (0,)
                    else:
                        cnn_fc_feat = (64, n_imgfeat,)
                    img_feature_tensor, cnn_weights, img_fc_weights = cnn_network(self.img_input, conv_sizes,
                                                                                  num_fc=cnn_fc_feat)
                    self.cnn_weights = cnn_weights
                    self.img_fc_weights = img_fc_weights

                    if n_imgfeat < 0:
                        self.image_features = tf.layers.flatten(img_feature_tensor[len(conv_sizes)])
                    else:
                        self.image_features = img_feature_tensor[-1]
                    self.full_feature = self.comb_method(self.st_enabled * self.state_input,
                                                         self.img_enabled[:, tf.newaxis] * self.image_features)

                else:
                    self.full_feature = self.state_input
                    self.cnn_weights = []
                    self.img_fc_weights = []
            hid1_size = (self.full_feature.shape[1].value) * 2
            hid3_size = action_shape[0] * 10
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            hidden_sizes = (hid1_size, hid2_size, hid3_size)
            logging.info("policy hidden sizes: {}".format(hidden_sizes))

            h1 = tf.layers.dense(self.full_feature, hidden_sizes[0], activation=tf.tanh,
                                 kernel_initializer=tf.orthogonal_initializer())
            h2 = tf.layers.dense(h1, hidden_sizes[1], activation=tf.tanh,
                                 kernel_initializer=tf.orthogonal_initializer())
            h3 = tf.layers.dense(h2, hidden_sizes[2], activation=tf.tanh,
                                 kernel_initializer=tf.orthogonal_initializer())
            self.action_dist_means_n = tf.layers.dense(h3, np.prod(action_shape), activation=tf.tanh,
                                                       kernel_initializer=tf.orthogonal_initializer(gain=0.1))

            # wd_dict = {}
            # h1 = tf.nn.tanh(
            #     dense(self.full_feature, 64, "h1", weight_init=U.normc_initializer(1.0), bias_init=0.0,
            #           weight_loss_dict=wd_dict))
            # h2 = tf.nn.tanh(
            #     dense(h1, 64, "h2", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
            # self.action_dist_means_n = dense(h2, np.prod(action_shape), "mean", weight_init=U.normc_initializer(0.1),
            #                                  bias_init=0.0,
            #                                  weight_loss_dict=wd_dict)  # Mean control output



            # logvar_speed = (10 * hid3_size) // 48
            # log_vars = self.log_vars = tf.get_variable("%s_logvars" % scope, (logvar_speed, action_shape[0]),
            #                                            tf.float32,
            #                                            tf.constant_initializer(0.0))
            # self.action_dist_logstds_n = tf.reduce_sum(log_vars, axis=0) + np.log(0.5)


            # self.action_dist_logstd_param = tf.Variable(
            #     initial_value=(np.log(1.0) + 0.001 * np.random.randn(*action_shape)).astype(np.float32),
            #     trainable=True, name="%spolicy_logstd" % scope)
            # self.action_dist_logstds_n = tf.tile(tf.expand_dims(self.action_dist_logstd_param, 0),
            #                                      [tf.shape(self.action_dist_means_n)[0], 1])
            # self.action_dist_logstds_n = self.action_dist_logstd_param


            self.action_dist_logstd_param = logstd_1a = tf.get_variable("logstd", action_shape, tf.float32,
                                                                        tf.zeros_initializer())  # Variance on outputs
            logstd_1a = tf.expand_dims(logstd_1a, 0)
            self.action_dist_logstds_n = tf.tile(logstd_1a, [tf.shape(self.action_dist_means_n)[0], 1])
            std_1a = tf.exp(logstd_1a)
            std_na = tf.tile(std_1a, [tf.shape(self.action_dist_means_n)[0], 1])

            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]
            if not cnn_trainable:
                self.var_list = [v for v in self.var_list if not (v in self.cnn_weights or v in self.img_fc_weights)]

        # log_std_var = tf.maximum(self.action_dist_logstds_n, np.log(self.min_std))
        batch_size = tf.shape(self.state_input)[0]
        self.batch_size_float = tf.cast(batch_size, tf.float32)
        # self.action_dist_log_stds_n = log_std_var  # self.net.action_dist_logstds_n  #
        # self.action_dist_std_n = tf.exp(self.action_dist_log_stds_n)
        self.old_dist_info_vars = dict(mean=self.old_dist_means_n, log_std=self.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.action_dist_means_n, log_std=self.action_dist_logstds_n)
        # self.likehood_action_dist = self.distribution.log_likelihood_sym(self.action_n, self.new_dist_info_vars)
        self.new_likelihood_sym = self.distribution.log_likelihood_sym(self.action_n, self.new_dist_info_vars)
        self.old_likelihood = self.distribution.log_likelihood_sym(self.action_n, self.old_dist_info_vars)

        self.ratio_n = tf.exp(self.new_likelihood_sym - self.old_likelihood)
        self.p_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list])
        self.PPO_eps = 0.2
        self.clipped_ratio = tf.clip_by_value(self.ratio_n, 1.0 - self.PPO_eps, 1.0 + self.PPO_eps)
        self.raw_surr = self.ratio_n * self.advant
        self.clipped_surr = self.clipped_ratio * self.advant
        surr = self.surr = -tf.reduce_mean(tf.minimum(self.raw_surr,
                                                      self.clipped_surr))  # Surrogate loss
        ac_dim = action_shape[0]
        ac_dist = tf.concat([tf.reshape(self.action_dist_means_n, [-1, ac_dim]), tf.reshape(std_na, [-1, ac_dim])], 1)
        logprob_n = - U.sum(tf.log(ac_dist[:, ac_dim:]), axis=1) - 0.5 * tf.log(2.0 * np.pi) * ac_dim - 0.5 * U.sum(
            tf.square(ac_dist[:, :ac_dim] - self.action_n) / (tf.square(ac_dist[:, ac_dim:])),
            axis=1)  # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)

        # kl = .5 * U.mean(tf.square(logprob_n - oldlogprob_n)) # Approximation of KL divergence between old policy used to generate actions, and new policy used to compute logprob_n
        self.trad_surr_loss = - U.mean(
            self.advant * logprob_n)  # Loss function that we'll differentiate to get the policy gradient
        self.mean_loglike = - U.mean(logprob_n)  # Sampled loss of the policy

        # self.trad_surr_loss = - tf.reduce_mean(self.new_likelihood_sym * self.advant)
        # self.mean_loglike = - tf.reduce_mean(self.new_likelihood_sym)

        kl = self.kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ents_fixed = self.distribution.entropy(self.old_dist_info_vars)
        ents_sym = self.distribution.entropy(self.new_dist_info_vars)  # - self.new_likelihood_sym
        ent = self.ent = tf.reduce_sum(ents_sym) / self.batch_size_float
        self.losses = [surr, kl, ent]
        self.get_flat_params = GetFlat(self.var_list, session=self.session)  # get theta from var_list
        self.set_params_with_flat_data = SetFromFlat(self.var_list, session=self.session)  # set theta from var_List
        # get g
        self.pg = flatgrad(surr, self.var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars) / self.batch_size_float
        grads = tf.gradients(kl_firstfixed, self.var_list)

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                           {self.state_input: obs[0],
                            self.img_input: obs[1]})
