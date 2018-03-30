import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
# from tensorflow.contrib.layers import initializers as tf_init
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
# import prettytensor as pt
import logging
from tf_utils import LbfgsOptimizer, run_batched, aggregate_feature, lrelu
from cnn import cnn_network
from scaling_orth import ScalingOrth
concat = np.concatenate


class MultiBaseline(object):
    coeffs = None

    def __init__(self, session=None, main_scope="value_f",
                 obs_space=None, n_imgfeat=1, activation=tf.nn.tanh,
                 max_iter=25, timestep_limit=1000, comb_method=aggregate_feature,
                 minibatch_size=256,
                 cnn_trainable=True,
                 f_build_cnn=None):
        self.session = session
        self.max_iter = max_iter
        self.l2_k = 1e-4
        self.mix_frac = 1
        self.timestep_limit = timestep_limit
        self.scope = main_scope
        self.minibatch_size = minibatch_size
        self.comb_method = comb_method
        self.normalize = True
        with tf.variable_scope(main_scope):
            # add  timestep
            self.state_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name="x")
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name='st_enabled')
            self.time_input = tf.placeholder(tf.float32, shape=(None, 1), name="t")

            self.sigma = tf.get_variable("scale_sigma", initializer=tf.constant(1.0), trainable=False)
            self.mu = tf.get_variable("scale_mu", initializer=tf.constant(0.0), trainable=False)
            self.new_std = tf.placeholder(tf.float32, shape=[], name="scale_sgma")
            self.new_mean = tf.placeholder(tf.float32, shape=[], name="scale_mu")

            self.y = tf.placeholder(tf.float32, shape=[None], name="y")
            self.final_state = self.st_enabled * self.state_input
            self.n_imgfeat = n_imgfeat
            if n_imgfeat != 0:
                with tf.variable_scope("img") as this_scope:
                    self.img_input = tf.placeholder(tf.uint8, shape=(None,) + obs_space[1].shape, name="img")
                    self.img_float = tf.cast(self.img_input, tf.float32) / 255
                    assert f_build_cnn is not None
                    img_net_layers, cnn_weights, img_fc_weights = f_build_cnn(self.img_float)
                    self.cnn_weights = cnn_weights
                    self.img_fc_weights = img_fc_weights

                    if n_imgfeat < 0:
                        self.image_features = tf.layers.flatten(img_net_layers[-1])
                    else:
                        self.image_features = img_net_layers[-1]
                    self.final_image_features = self.image_features

                    self.img_var_list = tf.trainable_variables(scope=this_scope.name)
                    self.img_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.img_var_list])
            else:
                self.img_var_list = []
                self.cnn_weights = []
                self.img_fc_weights = []
                self.final_image_features = None
            with tf.variable_scope("nethigher") as this_scope:
                if self.final_image_features is not None:
                    self.aggregated_feature = \
                        self.comb_method(self.final_state, self.img_enabled[:, tf.newaxis] * self.final_image_features)
                else:
                    self.aggregated_feature = self.final_state
                self.full_feature = tf.concat(
                    axis=1,
                    values=[self.aggregated_feature, self.time_input])

                hidden_sizes = (64, 64)
                logging.info("critic hidden sizes: {}".format(hidden_sizes))
                self.fc_layers = [self.full_feature]
                for hidden_size in hidden_sizes:
                    h = tf.layers.dense(self.fc_layers[-1], hidden_size,
                                        activation=activation, kernel_initializer=ScalingOrth())
                    self.fc_layers.append(h)
                with tf.variable_scope("final") as scope_last:
                    pre_y = tf.layers.dense(self.fc_layers[-1], 1, activation=None,
                                            kernel_initializer=ScalingOrth())
                    self.last_w, self.last_b = tf.trainable_variables(scope=scope_last.name)
                    y = pre_y * self.sigma + self.mu

                self.net = tf.reshape(y, (-1,))  # why reshape?
                err = self.y - self.net

                self.real_mse = tf.reduce_mean(err ** 2)
                self.norm_mse = self.real_mse / self.sigma ** 2
                self.mse = self.norm_mse if self.normalize else self.real_mse
                alpha = 1.0
                new_sigma = alpha * self.new_std + (1 - alpha) * self.sigma
                new_mu = alpha * self.new_mean + (1 - alpha) * self.mu
                w_update = tf.assign(self.last_w, new_sigma ** -1 * self.sigma * self.last_w)
                b_update = tf.assign(self.last_b, new_sigma ** -1 * (self.sigma * self.last_b + self.mu - new_mu))
                with tf.control_dependencies([w_update, b_update]):
                    sigma_update = tf.assign(self.sigma, self.new_std)
                    mu_update = tf.assign(self.mu, self.new_mean)
                self.scale_updates = [w_update, b_update, sigma_update, mu_update]
                self.st_var_list = tf.trainable_variables(this_scope.name)
                self.st_var_list = [i for i in self.st_var_list if i not in self.img_var_list]
            self.var_list = [*self.img_var_list, *self.st_var_list]
            if not cnn_trainable:
                self.var_list = [v for v in self.var_list if not (v in self.cnn_weights or v in self.img_fc_weights)]
            self.st_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.st_var_list])
            self.l2 = self.st_l2  #+ self.img_l2
            self.final_loss = self.mse  # S+ (self.l2) * self.l2_k

            self.lr = 1e-2 / np.sqrt(hidden_sizes[1])
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train = self.opt.minimize(self.final_loss, aggregation_method=tf.AggregationMethod.DEFAULT,
                                           var_list=self.var_list)

        self.debug_mode = True

    def explained_var(self, ypred, y):
        assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

    def print_loss(self, feed):
        mse = \
            run_batched([self.real_mse],
                        feed=feed, N=feed[self.y].shape[0],
                        session=self.session,
                        minibatch_size=self.minibatch_size)[0]
        # ypred = self.session.run(self.net, feed_dict=feed)
        # ex_var = self.explained_var(ypred, feed[self.y])
        # logging.debug("vf:\n mse:{}\texplained_var:{}".format(mse, ex_var))
        logging.debug("vf:\n mse:{}".format(mse))

    def fit(self, path_dict, update_mode="full", num_pass=1, pid=None):
        returns = path_dict["returns"]

        obj = returns
        state_mat = path_dict["state_input"]

        times = path_dict["times"]
        st_enabled = path_dict["st_enabled"]
        img_enabled = path_dict["img_enabled"]
        feed = {self.state_input: state_mat,
                self.time_input: times, self.y: obj,
                self.st_enabled: st_enabled, self.img_enabled: img_enabled}
        if self.n_imgfeat != 0:
            img_mat = path_dict["img_input"]
            feed[self.img_input] = img_mat
        batch_N = returns.shape[0]
        train_op = []
        if update_mode == "full":
            train_op = [self.train]
        elif update_mode == "both":
            train_op = [self.mse, self.img_train, self.upper_train]
        elif update_mode == "img":
            train_op = [self.img_train]
        elif update_mode == "st":
            train_op = [self.mse, self.upper_train]
        if self.debug_mode and (pid is None or pid == 0):
            logging.debug("before vf optimization")
            self.print_loss(feed)
        new_mu = obj.mean()
        new_sigma = obj.std()
        self.session.run(self.scale_updates, feed_dict={self.new_mean: new_mu,
                                                        self.new_std: new_sigma})

        for n_pass in range(num_pass):
            training_inds = np.random.permutation(batch_N)
            for start in range(0, batch_N, self.minibatch_size):
                if start > batch_N - 2 * self.minibatch_size:
                    end = batch_N
                else:
                    end = start + self.minibatch_size
                slc = training_inds[range(start, end)]
                this_feed = {k: v[slc] for (k, v) in list(feed.items())}
                self.session.run(train_op, feed_dict=this_feed)
                if end == batch_N:
                    break

        if self.debug_mode and (pid is None or pid == 0):
            logging.debug("after vf optimization")
            self.print_loss(feed)

    def predict(self, path):
        if self.net is None:
            raise ValueError("value net is None")
        else:
            feed = {self.state_input: path["observation"][0],
                    self.time_input: path["times"],
                    self.st_enabled: path["st_enabled"], self.img_enabled: path["img_enabled"]}
            if self.n_imgfeat != 0:
                feed[self.img_input] = path["observation"][1]
            ret = self.session.run(self.net, feed_dict=feed)
            return np.reshape(ret, (ret.shape[0],))
