import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
# from tensorflow.contrib.layers import initializers as tf_init
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
# import prettytensor as pt
import logging
from tf_utils import LbfgsOptimizer, run_batched, aggregate_feature, lrelu
from cnn import cnn_network
concat = np.concatenate


# TODO: l_bfgs optimizer
class MultiBaseline(object):
    coeffs = None

    def __init__(self, session=None, scope="value_f",
                 obs_space=None,
                 conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)), n_imgfeat=1, activation=tf.nn.tanh,
                 max_iter=25, timestep_limit=1000, comb_method=aggregate_feature,
                 cnn_trainable=True):
        self.session = session
        self.max_iter = max_iter
        self.use_lbfgs_b = False  # not with_image
        self.l2_k = 1e-4
        self.mix_frac = 1
        self.timestep_limit = timestep_limit
        self.scope = scope
        self.minibatch_size = 256
        self.comb_method = comb_method
        with tf.variable_scope(scope):
            # add  timestep
            self.state_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name="x")
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name='st_enabled')
            self.time_input = tf.placeholder(tf.float32, shape=(None, 1), name="t")
            self.y = tf.placeholder(tf.float32, shape=[None], name="y")
            self.final_state = self.st_enabled * self.state_input
        img_scope = "img_" + scope  # + "_img"
        self.n_imgfeat = n_imgfeat
        if n_imgfeat != 0:
            with tf.variable_scope(scope):
                self.img_input = tf.placeholder(tf.uint8, shape=(None,) + obs_space[1].shape, name="img")
                self.img_float = tf.cast(self.img_input, tf.float32) / 255
            with tf.variable_scope(img_scope):

                if n_imgfeat < 0:
                    cnn_fc_feat = (0,)
                else:
                    cnn_fc_feat = (64, n_imgfeat,)
                img_feature_tensor, cnn_weights, img_fc_weights = cnn_network(self.img_float, conv_sizes,
                                                                              num_fc=cnn_fc_feat)
                self.cnn_weights = cnn_weights
                self.img_fc_weights = img_fc_weights

                if n_imgfeat < 0:
                    self.image_features = tf.layers.flatten(img_feature_tensor[len(conv_sizes)])
                else:
                    self.image_features = img_feature_tensor[-1]
                self.final_image_features = self.image_features

                self.img_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=img_scope)
                self.img_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.img_var_list])
                # self.img_loss = tf.reduce_mean(tf.square(self.pre_image_features[0] - self.state_input[:, :]))
                # self.pretrain_loss = self.img_loss
                # self.img_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
                # self.img_train = self.img_opt.minimize(self.pretrain_loss, aggregation_method=tf.AggregationMethod.DEFAULT,
                #                                        var_list=self.img_var_list)
        else:
            self.img_var_list = []
            self.cnn_weights = []
            self.img_fc_weights = []
            self.final_image_features = tf.constant(0.0)
        with tf.variable_scope(scope):
            self.aggregated_feature = \
                self.comb_method(self.final_state, self.img_enabled[:, tf.newaxis] * self.final_image_features)
            self.full_feature = tf.concat(
                axis=1,
                values=[self.aggregated_feature, self.time_input])

            # hid1_size = (self.full_feature.shape[1].value) * 2
            # hid3_size = 5
            # hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # hidden_sizes = (hid1_size, hid2_size, hid3_size)
            hidden_sizes = (64, 64)
            logging.info("critic hidden sizes: {}".format(hidden_sizes))
            self.fc_layers = [self.full_feature]
            for hidden_size in hidden_sizes:
                h = tf.layers.dense(self.fc_layers[-1], hidden_size,
                                    activation=activation, kernel_initializer=tf.orthogonal_initializer())
                self.fc_layers.append(h)
            # hidden_units = pt.wrap(self.full_feature).sequential()
            # for hidden_size in hidden_sizes:
            #     hidden_units.fully_connected(hidden_size, activation_fn=activation,
            #                                  weights=tf.orthogonal_initializer()
            #                                  )
            y = tf.layers.dense(self.fc_layers[-1], 1, activation=None, kernel_initializer=tf.orthogonal_initializer())

            self.net = tf.reshape(y, (-1,))  # why reshape?
            self.mse = tf.reduce_mean(tf.square(self.net - self.y))
            self.st_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.st_var_list = [i for i in self.st_var_list if i not in self.img_var_list]
            self.var_list = [*self.img_var_list, *self.st_var_list]
            if not cnn_trainable:
                self.var_list = [v for v in self.var_list if not (v in self.cnn_weights or v in self.img_fc_weights)]
            self.st_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.st_var_list])
            self.l2 = self.st_l2  #+ self.img_l2
            self.final_loss = self.mse  # S+ (self.l2) * self.l2_k

            if self.use_lbfgs_b:
                self.opt = LbfgsOptimizer(
                    self.final_loss, params=self.st_var_list, maxiter=self.max_iter, session=self.session)
                # self.optimizer = ScipyOptimizerInterface(loss=self.final_loss, method="L-BFGS-B",
                #                                          options={'maxiter': self.max_iter}
                #                                          )
                self.upper_train = None
                self.train = None
            else:
                self.lr = 1e-2 / np.sqrt(hidden_sizes[1])
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.train = self.opt.minimize(self.final_loss, aggregation_method=tf.AggregationMethod.DEFAULT,
                                               var_list=self.var_list)
                # self.train = None
                # self.upper_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
                # self.upper_opt = tf.train.AdagradOptimizer(learning_rate=0.0005,initial_accumulator_value=0.1)
                # self.upper_train = self.upper_opt.minimize(self.final_loss,
                #                                            aggregation_method=tf.AggregationMethod.DEFAULT,
                #                                            var_list=self.st_var_list)
        self.session.run(tf.global_variables_initializer())
        self.debug_mode = True

    def explained_var(self, ypred, y):
        assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary

    # def _features(self, path):
    #     obs = path["observations"]
    #     # l = (path["observations"].shape[0])
    #     # al = np.arange(l).reshape(-1, 1) / 10.0
    #     ret = np.concatenate((obs, path["times"][:, None],), axis=1)
    #     return ret
    def print_loss(self, feed):
        # mse, l2, img_loss, img_l2 = \
        #     run_batched([self.mse, self.l2, self.img_loss, self.img_l2],
        #                 feed=feed, N=feed[self.y].shape[0],
        #                 session=self.session,
        #                 minibatch_size=self.minibatch_size)
        # logging.debug("vf:\n mse={}\tl2={}\nimg_loss={}\nimg_l2={}\n".format(mse, l2, img_loss, img_l2))
        mse = \
            run_batched([self.mse],
                        feed=feed, N=feed[self.y].shape[0],
                        session=self.session,
                        minibatch_size=self.minibatch_size)[0]
        ypred = self.session.run(self.net, feed_dict=feed)
        ex_var = self.explained_var(ypred, feed[self.y])
        logging.debug("vf:\n mse:{}\texplained_var:{}".format(mse, ex_var))

    def fit(self, path_dict, update_mode="full", num_pass=1):
        # featmat = self._features(paths)
        # returns = paths["values"]
        returns = path_dict["returns"]
        # if self.mix_frac != 1:
        #     obj = returns * self.mix_frac + self.predict(paths) * (1 - self.mix_frac)
        # else:
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
        if self.debug_mode:
            logging.debug("before vf optimization")
            self.print_loss(feed)


        if self.use_lbfgs_b:
            # self.optimizer.minimize(session=self.session,
            #                         feed_dict=feed)
            self.opt.update(session=self.session, feed=feed)
        else:
            for n_pass in range(num_pass):
                training_inds = np.random.permutation(batch_N)
                for start in range(0, batch_N, self.minibatch_size):  # TODO: verify this
                    if start > batch_N - 2 * self.minibatch_size:
                        end = batch_N
                    else:
                        end = start + self.minibatch_size
                    slc = training_inds[range(start, end)]
                    this_feed = {k: v[slc] for (k, v) in list(feed.items())}
                    self.session.run(train_op, feed_dict=this_feed)
                    if end == batch_N:
                        break

        if self.debug_mode:
            logging.debug("after vf optimization")
            self.print_loss(feed)

    def predict(self, path):
        if self.net is None:
            raise ValueError("value net is None")
            # return np.zeros((path["values"].shape[0]))
        else:
            # ret = self.session.run(self.net, {self.x: self._features(path)})
            feed = {self.state_input: path["observation"][0],
                    self.time_input: path["times"],
                    self.st_enabled: path["st_enabled"], self.img_enabled: path["img_enabled"]}
            if self.n_imgfeat != 0:
                feed[self.img_input] = path["observation"][1]
            ret = self.session.run(self.net, feed_dict=feed)
            return np.reshape(ret, (ret.shape[0],))
