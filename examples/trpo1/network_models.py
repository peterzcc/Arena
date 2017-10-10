from __future__ import absolute_import
import tensorflow as tf
# from tensorflow.contrib.layers import initializers as tf_init
from tensorflow.contrib.layers import variance_scaling_initializer
import prettytensor as pt
import numpy as np
from tf_utils import aggregate_feature, lrelu
dtype = tf.float32





class MultiNetwork(object):
    def __init__(self, scope, observation_space, action_shape,
                 conv_sizes=(((4, 4), 16, 2), ((4, 4), 16, 1)),
                 n_imgfeat=1, extra_feaatures=[], st_enabled=None, img_enabled=None,
                 comb_method=aggregate_feature):
        self.comb_method = comb_method
        with tf.variable_scope("%s_shared" % scope):
            self.state_input = tf.placeholder(
                dtype, shape=(None,) + observation_space[0].shape, name="%s_state" % scope)
            if n_imgfeat > 0:
                self.img_input = \
                    tf.placeholder(tf.float32, shape=(None,) + observation_space[1].shape, name="%s_img" % scope)

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
                # self.full_feature = tf.concat(axis=1, values=[self.st_enabled * self.state_input. *extra_feaatures])
                self.full_feature = self.comb_method(self.st_enabled * self.state_input, extra_feaatures[0])
            else:
                if n_imgfeat > 0:
                    expanded_img = self.img_input  # tf.expand_dims(self.img_input, -1)
                    img_features = pt.wrap(expanded_img).sequential()
                    for conv_size in conv_sizes:
                        img_features.conv2d(conv_size[0], depth=conv_size[1], activation_fn=lrelu,
                                            stride=conv_size[2],
                                            weights=variance_scaling_initializer(factor=0.1,
                                                                                 mode='FAN_AVG',
                                                                                 uniform=True)
                                            )
                    img_features.flatten()
                    img_features.fully_connected(n_imgfeat, activation_fn=tf.nn.tanh,
                                                 weights=variance_scaling_initializer(factor=0.1,
                                                                                      mode='FAN_AVG',
                                                                                      uniform=True)
                                                 )
                    self.image_features = img_features.as_layer()
                    self.full_feature = self.comb_method(self.st_enabled * self.state_input,
                                                         self.img_enabled[:, tf.newaxis] * self.image_features)
                    # self.full_feature = self.st_enabled * self.state_input + \
                    #                     self.img_enabled[:, tf.newaxis] * self.image_features
                    # self.full_feature = self.image_features
                else:
                    self.full_feature = self.state_input
            hid1_size = observation_space[0].shape[0] * 10
            hid3_size = action_shape[0] * 10
            hid2_size = int(np.sqrt(hid1_size * hid3_size))

            self.action_dist_means_n = (pt.wrap(self.full_feature).
                                        fully_connected(hid1_size, activation_fn=tf.tanh,
                                                        weights=variance_scaling_initializer(factor=0.1,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                                        name="%s_fc1" % scope).
                                        fully_connected(hid2_size, activation_fn=tf.tanh,
                                                        weights=variance_scaling_initializer(factor=0.1,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                                        name="%s_fc1" % scope).
                                        fully_connected(hid3_size, activation_fn=tf.tanh,
                                                        weights=variance_scaling_initializer(factor=0.1,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                                        name="%s_fc2" % scope).
                                        fully_connected(np.prod(action_shape),
                                                        activation_fn=None,
                                                        weights=variance_scaling_initializer(factor=0.1,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                                        name="%s_fc3" % scope))
            logvar_speed = (10 * hid3_size) // 48
            log_vars = tf.get_variable("%s_logvars" % scope, (logvar_speed, action_shape[0]), tf.float32,
                                       tf.constant_initializer(0.0))
            self.action_dist_logstds_n = tf.reduce_sum(log_vars, axis=0) + np.log(0.5)

            # self.action_dist_logstd_param = tf.Variable(
            #     initial_value=(np.log(0.7) + 0.001 * np.random.randn(1, *action_shape)).astype(np.float32),
            #     trainable=True, name="%spolicy_logstd" % scope)

            # self.action_dist_logstds_n = tf.tile(self.action_dist_logstd_param,
            #                                      tf.stack((tf.shape(self.action_dist_means_n)[0], 1)))
            self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                           {self.state_input: obs[0],
                            self.img_input: obs[1]})
