import tensorflow as tf
import numpy as np


class ProbType(object):
    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        pass

    def likelihood_ratio_sym(self, x_var, new_dist_info_vars, old_dist_info_vars):
        """
        \frac{\pi_\theta}{\pi_{old}}
        """
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        pass

    def kl_sym_firstfixed(self, old_dist_info_vars):
        pass

    def sample(self, dist_info):
        pass

    def log_likelihood(self, xs, dist_info):
        pass

    def entropy(self, dist_info):
        pass

    def create_dist_vars(self, last_layer, dtype=tf.float32):
        pass


class Categorical(ProbType):
    def __init__(self, num_cat):
        self.n = num_cat

    def create_dist_vars(self, last_layer, dtype=tf.float32):
        old_logits = tf.placeholder(dtype,
                                    shape=(None, self.n), name="old_logits")
        logits = tf.layers.dense(last_layer, self.n,
                                 kernel_initializer=tf.variance_scaling_initializer(
                                     scale=1.0, mode="fan_avg", distribution="normal", dtype=dtype))
        old_dist_vars = dict(logits=old_logits)
        dist_vars = dict(logits=logits)
        sample = tf.distributions.Categorical(logits=logits).sample()
        return dist_vars, old_dist_vars, sample, {}

    def log_likelihood_sym(self, x_var, dist_info_vars):
        one_hot_actions = tf.one_hot(x_var, self.n)
        logp = -tf.nn.softmax_cross_entropy_with_logits(logits=dist_info_vars["logits"], labels=one_hot_actions)
        return logp

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_l = old_dist_info_vars["logits"]
        new_l = new_dist_info_vars["logits"]
        old_dist = tf.distributions.Categorical(logits=old_l)
        new_dist = tf.distributions.Categorical(logits=new_l)
        kl = old_dist.kl_divergence(new_dist)
        return kl

    def entropy(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.entropy()

    def sample(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.sample()

    def kf_loglike(self, action_n, dist_vars, interm_vars):
        return self.log_likelihood_sym(action_n, dist_vars)


class DiagonalGaussian(ProbType):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def create_dist_vars(self, last_layer, dtype=tf.float32):
        old_mean_n = tf.placeholder(dtype, shape=(None, self.dim),
                                    name="oldaction_dist_means")
        old_logstd_n = tf.placeholder(dtype, shape=(None, self.dim),
                                      name="oldaction_dist_logstds")
        mean_n = tf.layers.dense(last_layer, self.dim,  # activation=tf.tanh,
                                 kernel_initializer=tf.variance_scaling_initializer(
                                     scale=1.0, mode="fan_avg", distribution="normal", dtype=dtype))

        logstd_param = logstd_1a = tf.get_variable("logstd", (self.dim,), tf.float32,
                                                   tf.zeros_initializer())  # Variance on outputs
        logstd_1a = tf.expand_dims(logstd_1a, 0)
        std_1a = tf.exp(logstd_1a)
        logstd_n = tf.tile(logstd_1a, [tf.shape(mean_n)[0], 1])
        std_n = tf.tile(std_1a, [tf.shape(mean_n)[0], 1])
        old_dist_vars = dict(mean=old_mean_n, logstd=old_logstd_n)
        dist_vars = dict(mean=mean_n, logstd=logstd_n)
        interm_vars = dict(std=std_n)
        sample = tf.distributions.Normal(loc=mean_n, scale=std_n).sample()
        return dist_vars, old_dist_vars, sample, interm_vars

    def gen_dist_info(self, mean, log_std):
        return dict(mean=mean, log_std=log_std)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["logstd"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["logstd"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, -1)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        \frac{1}{(2\pi)^{\frac{n}{2}}\sigma_\theta}exp(-(\frac{a-\mu_{\pi_\theta}}{2\sigma_\theta})^2)
        :param x_var:
        :param dist_info_vars:
        :return:
        """
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["logstd"]
        zs = (x_var - means) * tf.exp(-log_stds)
        return - tf.reduce_sum(log_stds, -1) - \
               0.5 * tf.reduce_sum(tf.square(zs), -1) - \
               0.5 * means.get_shape()[-1].value * np.log(2 * np.pi)

    def kl_sym_firstfixed(self, old_dist_info_vars):
        mu = old_dist_info_vars["mean"]
        logstd = old_dist_info_vars["logstd"]
        mu1, logstd1 = tuple(map(tf.stop_gradient, [mu, logstd]))
        mu2, logstd2 = mu, logstd

        return self.kl_sym(dict(mean=mu1, log_std=logstd1), dict(mean=mu2, log_std=logstd2))

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["logstd"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["logstd"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["logstd"]
        return tf.reduce_sum(log_stds, 1) + 0.5 * np.log(2 * np.pi * np.e) * self.dim

    def kf_loglike(self, action_n, dist_vars, interm_vars):
        mean_n = dist_vars["mean"]
        std_n = interm_vars["std"]
        logprob_n = - tf.reduce_sum(tf.log(std_n), axis=1) - 0.5 * tf.log(
            2.0 * np.pi) * self.dim - 0.5 * tf.reduce_sum(
            tf.square(mean_n - action_n) / (tf.square(std_n)),
            axis=1)  # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
        return logprob_n

    @property
    def dist_info_keys(self):
        return ["mean", "logstd"]