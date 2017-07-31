import tensorflow as tf
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
# from tensorflow.contrib.layers import initializers as tf_init
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import prettytensor as pt
import logging
from tf_utils import LbfgsOptimizer, run_batched, aggregate_feature, lrelu

concat = np.concatenate


# TODO: l_bfgs optimizer
class MultiBaseline(object):
    coeffs = None

    def __init__(self, session=None, scope="value_f",
                 obs_space=None, hidden_sizes=(64, 64),
                 conv_sizes=(((4, 4), 16, 2), ((3, 3), 16, 1)), n_imgfeat=1, activation=tf.nn.tanh,
                 max_iter=25, timestep_limit=1000, with_image=True, comb_method=aggregate_feature):
        self.session = session
        self.max_iter = max_iter
        self.use_lbfgs_b = False  # not with_image
        self.l2_k = 1e-4
        self.mix_frac = 1
        self.timestep_limit = timestep_limit
        self.scope = scope
        self.minibatch_size = 32
        self.comb_method = comb_method
        assert len(obs_space) == 2
        with tf.variable_scope(scope):
            # add  timestep
            self.state_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name="x")
            self.img_input = tf.placeholder(tf.float32, shape=(None,) + obs_space[1].shape, name="img")
            self.img_enabled = tf.placeholder(tf.float32, shape=(None,), name='img_enabled')
            self.st_enabled = tf.placeholder(tf.float32, shape=(None,) + obs_space[0].shape, name='st_enabled')
            self.time_input = tf.placeholder(tf.float32, shape=(None, 1), name="t")
            self.y = tf.placeholder(tf.float32, shape=[None], name="y")
            self.final_state = self.st_enabled * self.state_input
        img_scope = "img_" + scope  # + "_img"
        with tf.variable_scope(img_scope):
            expanded_img = self.img_input  # tf.expand_dims(self.img_input, -1)
            img_features = pt.wrap(expanded_img).sequential()
            for conv_size in conv_sizes:
                img_features.conv2d(conv_size[0], depth=conv_size[1], activation_fn=lrelu,
                                    stride=conv_size[2],
                                    weights=variance_scaling_initializer(factor=1.0,
                                                                         mode='FAN_AVG',
                                                                         uniform=True)
                                    )
            img_features.flatten()
            img_features.fully_connected(16, activation_fn=lrelu,
                                         weights=variance_scaling_initializer(factor=1.0,
                                                                              mode='FAN_AVG',
                                                                              uniform=True)
                                         )
            img_features.fully_connected(n_imgfeat, activation_fn=None,
                                         weights=variance_scaling_initializer(factor=1.0,
                                                                              mode='FAN_AVG',
                                                                              uniform=True)
                                         )

            # img_features.flatten()
            self.pre_image_features = [img_features.as_layer()]
            self.image_features = [self.img_enabled[:, tf.newaxis] * self.pre_image_features[0]]
            self.img_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=img_scope)
            self.img_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.img_var_list])
            self.img_loss = tf.reduce_mean(tf.square(self.pre_image_features[0] - self.state_input[:, :]))
            self.pretrain_loss = self.img_loss
            self.img_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.img_train = self.img_opt.minimize(self.pretrain_loss, aggregation_method=tf.AggregationMethod.DEFAULT,
                                                   var_list=self.img_var_list)

        with tf.variable_scope(scope):
            if with_image:
                self.full_feature = tf.concat(
                    axis=1,
                    values=[self.final_state, *self.image_features, self.time_input])
            else:
                self.aggregated_feature = self.comb_method(self.final_state, self.image_features[0])
                self.full_feature = tf.concat(
                    axis=1,
                    values=[self.aggregated_feature, self.time_input])
            hidden_units = pt.wrap(self.full_feature).sequential()
            for hidden_size in hidden_sizes:
                hidden_units.fully_connected(hidden_size, activation_fn=activation,
                                             weights=variance_scaling_initializer(factor=1.0,
                                                                                  mode='FAN_AVG',
                                                                                  uniform=True)
                                             )

            self.net = tf.reshape(hidden_units.fully_connected(1).as_layer(), (-1,))  # why reshape?
            self.mse = tf.reduce_mean(tf.square(self.net - self.y))
            self.st_var_list = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.st_var_list = [i for i in self.st_var_list if i not in self.img_var_list]
            self.var_list = [*self.img_var_list, *self.st_var_list]
            self.st_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.st_var_list])
            self.l2 = self.st_l2  #+ self.img_l2
            self.final_loss = self.mse + (self.l2) * self.l2_k

            if self.use_lbfgs_b:
                self.opt = LbfgsOptimizer(
                    self.final_loss, params=self.st_var_list, maxiter=self.max_iter, session=self.session)
                # self.optimizer = ScipyOptimizerInterface(loss=self.final_loss, method="L-BFGS-B",
                #                                          options={'maxiter': self.max_iter}
                #                                          )
                self.upper_train = None
                self.train = None
            else:
                self.opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
                self.train = self.opt.minimize(self.final_loss, aggregation_method=tf.AggregationMethod.DEFAULT,
                                               var_list=self.var_list)
                # self.train = None
                self.upper_opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
                # self.upper_opt = tf.train.AdagradOptimizer(learning_rate=0.0005,initial_accumulator_value=0.1)
                self.upper_train = self.upper_opt.minimize(self.final_loss,
                                                           aggregation_method=tf.AggregationMethod.DEFAULT,
                                                           var_list=self.st_var_list)
        self.session.run(tf.global_variables_initializer())
        self.debug_mode = True

    # def _features(self, path):
    #     obs = path["observations"]
    #     # l = (path["observations"].shape[0])
    #     # al = np.arange(l).reshape(-1, 1) / 10.0
    #     ret = np.concatenate((obs, path["times"][:, None],), axis=1)
    #     return ret
    def print_loss(self, feed):
        mse, l2, img_loss, img_l2 = \
            run_batched([self.mse, self.l2, self.img_loss, self.img_l2],
                        feed=feed, N=feed[self.y].shape[0],
                        session=self.session,
                        minibatch_size=self.minibatch_size)
        logging.debug("vf:\n mse={}\tl2={}\nimg_loss={}\nimg_l2={}\n".format(mse, l2, img_loss, img_l2))

        # mse = self.session.run([self.mse], feed_dict=feed)[0]
        # logging.debug("vf:\n mse:{}\n".format(mse))

    def fit(self, path_dict, update_mode="full", num_pass=1):
        # featmat = self._features(paths)
        # returns = paths["values"]
        returns = path_dict["returns"]
        # if self.mix_frac != 1:
        #     obj = returns * self.mix_frac + self.predict(paths) * (1 - self.mix_frac)
        # else:
        obj = returns
        state_mat = path_dict["state_input"]
        img_mat = path_dict["img_input"]
        times = path_dict["times"]
        st_enabled = path_dict["st_enabled"]
        img_enabled = path_dict["img_enabled"]

        feed = {self.state_input: state_mat, self.img_input: img_mat,
                self.time_input: times, self.y: obj,
                self.st_enabled: st_enabled, self.img_enabled: img_enabled}
        batch_N = returns.shape[0]
        if update_mode == "full":
            train_op = [self.mse, self.train]
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
                    this_feed = {self.state_input: state_mat[slc], self.img_input: img_mat[slc],
                                 self.time_input: times[slc], self.y: obj[slc],
                                 self.st_enabled: st_enabled[slc], self.img_enabled: img_enabled[slc]}
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
            feed = {self.state_input: path["observation"][0], self.img_input: path["observation"][1],
                    self.time_input: path["times"],
                    self.st_enabled: path["st_enabled"], self.img_enabled: path["img_enabled"]}
            ret = self.session.run(self.net, feed_dict=feed)
            return np.reshape(ret, (ret.shape[0],))
