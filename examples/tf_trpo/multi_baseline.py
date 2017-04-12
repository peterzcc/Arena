import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from tensorflow.contrib.layers import initializers as tf_init
import numpy as np
import prettytensor as pt
import logging
from tf_utils import LbfgsOptimizer

concat = np.concatenate


# TODO: l_bfgs optimizer
class MultiBaseline(object):
    coeffs = None

    def __init__(self, session=None, scope="value_f",
                 obs_space=None, hidden_sizes=(64, 64),
                 conv_sizes=(((4, 4), 16, 2), ((4, 4), 16, 1)), activation=tf.nn.tanh,
                 max_iter=25, timestep_limit=1000, with_image=True):
        self.session = session
        self.max_iter = max_iter
        self.use_lbfgs_b = True
        self.l2_k = 1e-3
        self.mix_frac = 1
        self.timestep_limit = timestep_limit
        self.scope = scope
        assert len(obs_space) == 2

        with tf.variable_scope(scope):
            # add  timestep
            self.state_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name="x")
            self.img_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[1].shape, name="img")

            self.time_input = tf.placeholder(tf.float32, shape=(None, 1), name="t")
            self.y = tf.placeholder(tf.float32, shape=[None], name="y")

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
                self.full_feature = tf.concat(
                    concat_dim=1,
                    values=[self.state_input, img_features.as_layer(), self.time_input])
            else:
                self.full_feature = tf.concat(
                    concat_dim=1,
                    values=[self.state_input, self.time_input])
            hidden_units = pt.wrap(self.full_feature).sequential()
            for hidden_size in hidden_sizes:
                hidden_units.fully_connected(hidden_size, activation_fn=activation,
                                             init=tf_init.variance_scaling_initializer(factor=1.0,
                                                                                       mode='FAN_AVG',
                                                                                       uniform=True)
                                             )

            self.net = tf.reshape(hidden_units.fully_connected(1).as_layer(), (-1,))  # why reshape?
            self.mse = tf.reduce_mean(tf.square(self.net - self.y))
            self.var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.var_list])
            self.final_loss = self.mse + self.l2 * self.l2_k
            if self.use_lbfgs_b:
                self.opt = LbfgsOptimizer(
                    self.final_loss, params=self.var_list, maxiter=self.max_iter, session=self.session)
                # self.optimizer = ScipyOptimizerInterface(loss=self.final_loss, method="L-BFGS-B",
                #                                          options={'maxiter': self.max_iter}
                #                                          )
            else:
                self.train = tf.train.AdamOptimizer().minimize(self.final_loss)
        self.session.run(tf.initialize_all_variables())
        self.debug_mode = True

    # def _features(self, path):
    #     obs = path["observations"]
    #     # l = (path["observations"].shape[0])
    #     # al = np.arange(l).reshape(-1, 1) / 10.0
    #     ret = np.concatenate((obs, path["times"][:, None],), axis=1)
    #     return ret

    def fit(self, paths):
        # featmat = self._features(paths)
        # returns = paths["values"]
        state_mat = concat([path["observation"][0] for path in paths], axis=0)
        img_mat = concat([path["observation"][1] for path in paths], axis=0)
        times = concat([path["times"] for path in paths], axis=0)
        returns = concat([path["return"] for path in paths])
        if self.mix_frac != 1:
            obj = returns * self.mix_frac + self.predict(paths) * (1 - self.mix_frac)
        else:
            obj = returns
        feed = {self.state_input: state_mat, self.img_input: img_mat,
                self.time_input: times, self.y: obj}
        if self.use_lbfgs_b:
            if self.debug_mode:
                mse, l2 = self.session.run([self.mse, self.l2], feed_dict=feed)
                logging.debug("vf_before: mse={}\tl2={}\n".format(mse, l2))

            # self.optimizer.minimize(session=self.session,
            #                         feed_dict=feed)
            self.opt.update(session=self.session, feed=feed)
            if self.debug_mode:
                mse, l2 = self.session.run([self.mse, self.l2], feed_dict=feed)
                logging.debug("vf_after: mse={}\tl2={}\n".format(mse, l2))

        else:
            for _ in range(self.max_iter):  # TODO: verify this
                loss, _ = self.session.run([self.mse, self.train], feed_dict=feed)

    def predict(self, path):
        if self.net is None:
            raise ValueError("value net is None")
            # return np.zeros((path["values"].shape[0]))
        else:
            # ret = self.session.run(self.net, {self.x: self._features(path)})
            feed = {self.state_input: path["observation"][0], self.img_input: path["observation"][1],
                    self.time_input: path["times"]}
            ret = self.session.run(self.net, feed_dict=feed)
            return np.reshape(ret, (ret.shape[0],))
