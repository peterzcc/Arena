from __future__ import absolute_import
import tensorflow as tf
from tensorflow.contrib.layers import initializers as tf_init
import prettytensor as pt
import numpy as np

dtype = tf.float32

# TODO: remove this class
class NetworkContinous(object):
    def __init__(self, scope, obs_shape, action_shape,
                 image_input=False):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                dtype, shape=(None,) + obs_shape, name="%s_obs" % scope)
            self.action_n = tf.placeholder(dtype, shape=(None,) + action_shape, name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)

            self.old_dist_means_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                     name="%s_oldaction_dist_logstds" % scope)

            self.action_dist_means_n = (pt.wrap(self.obs).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc1" % scope).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc2" % scope).
                                        fully_connected(np.prod(action_shape),
                                                        init=tf_init.variance_scaling_initializer(factor=0.01,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc3" % scope))
            # TODO: STD should be trainable, learn this later
            # TODO: understand this machine code, could be potentially prone to bugs
            self.action_dist_logstd_param = tf.Variable(
                initial_value=(np.log(1) + 0.01 * np.random.randn(1, *action_shape)).astype(np.float32),
                trainable=True, name="%spolicy_logstd" % scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                           {self.obs: obs})


class MultiNetwork(object):
    def __init__(self, scope, observation_space, action_shape, conv_sizes=(((4, 4), 16, 2), ((4, 4), 16, 1)),
                 with_image=True):
        with tf.variable_scope("%s_shared" % scope):
            self.state_input = tf.placeholder(
                dtype, shape=(None,) + observation_space[0].shape, name="%s_state" % scope)
            self.img_input = \
                tf.placeholder(tf.float32, shape=(None,) + observation_space[1].shape, name="%s_img" % scope)

            self.action_n = tf.placeholder(dtype, shape=(None,) + action_shape, name="%s_action" % scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant" % scope)

            self.old_dist_means_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                   name="%s_oldaction_dist_means" % scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=(None,) + action_shape,
                                                     name="%s_oldaction_dist_logstds" % scope)

            if with_image:
                expanded_img = tf.expand_dims(self.img_input, -1)
                img_features = pt.wrap(expanded_img).sequential()
                for conv_size in conv_sizes:
                    img_features.conv2d(conv_size[0], depth=conv_size[1], activation_fn=tf.nn.relu,
                                        stride=conv_size[2],
                                        init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                  mode='FAN_AVG',
                                                                                  uniform=True)
                                        )
                img_features.flatten()
                img_features.fully_connected(1, activation_fn=tf.nn.tanh,
                                             init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                       mode='FAN_AVG',
                                                                                       uniform=True)
                                             )

                # img_features.flatten()
                self.full_feature = tf.concat(
                    concat_dim=1,
                    values=[self.state_input, img_features.as_layer()])
            else:
                self.full_feature = self.state_input

            self.action_dist_means_n = (pt.wrap(self.full_feature).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc1" % scope).
                                        fully_connected(64, activation_fn=tf.nn.tanh,
                                                        init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc2" % scope).
                                        fully_connected(np.prod(action_shape),
                                                        init=tf_init.variance_scaling_initializer(factor=0.01,
                                                                                                  mode='FAN_AVG',
                                                                                                  uniform=True),
                                                        name="%s_fc3" % scope))
            # TODO: STD should be trainable, learn this later
            # TODO: understand this machine code, could be potentially prone to bugs
            self.action_dist_logstd_param = tf.Variable(
                initial_value=(np.log(1) + 0.01 * np.random.randn(1, *action_shape)).astype(np.float32),
                trainable=True, name="%spolicy_logstd" % scope)
            self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
                                                 tf.pack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                           {self.state_input: obs[0],
                            self.img_input: obs[1]})
