import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
from tensorflow.contrib.layers import initializers as tf_init
import numpy as np
import prettytensor as pt


# TODO: l_bfgs optimizer
class Baseline(object):
    coeffs = None

    def __init__(self, session=None, scope="value_f",
                 shape=None, hidden_sizes=(64, 64), activation=tf.nn.tanh,
                 max_iter=25):
        self.session = session
        self.max_iter = max_iter
        self.use_lbfgs_b = True
        self.l2_k = 1e-3
        self.mix_frac = 0.1

        with tf.variable_scope(scope):
            # add  timestep
            self.x = tf.placeholder(tf.float32, shape=(None, shape[0] + 1), name="x")
            self.y = tf.placeholder(tf.float32, shape=[None], name="y")
            hidden_units = pt.wrap(self.x).sequential()
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
                self.optimizer = ScipyOptimizerInterface(loss=self.final_loss, method="L-BFGS-B",
                                                         options={'maxiter': self.max_iter}
                                                         )
            else:
                self.train = tf.train.AdamOptimizer().minimize(self.final_loss)
        self.session.run(tf.initialize_all_variables())

    def _features(self, path):
        obs = path["observations"]
        l = (path["observations"].shape[0])
        # al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate((obs, path["times"][:, None],), axis=1)
        return ret

    def fit(self, paths):
        featmat = self._features(paths)
        returns = paths["values"]

        if self.use_lbfgs_b:
            obj = returns * self.mix_frac + self.predict(paths)
            self.optimizer.minimize(session=self.session,
                                    feed_dict={self.x: featmat, self.y: obj})
        else:
            for _ in range(self.max_iter):  # TODO: verify this
                loss, _ = self.session.run([self.mse, self.train], {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros((path["values"].shape[0]))
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0],))
