import tensorflow as tf
import numpy as np
from tf_utils import scale_positive_gradient_op

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

    def sample(self, dist_info):
        pass

    def log_likelihood(self, xs, dist_info):
        pass

    def entropy(self, dist_info):
        pass

    def create_dist_vars(self, last_layer, dtype=tf.float32):
        pass

    def regulation_loss(self, dist_info):
        return 0.0


_EPSILON = 1e-6


def categorical_crossentropy(output, labels, from_logits=False):
    """Categorical crossentropy with integer targets.
    # Arguments
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        target: An integer tensor.
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        epsilon = tf.constant(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        logits = tf.log(output)
    else:
        logits = None

    # output_shape = output.get_shape()
    # logits = tf.reshape(output, [-1, int(output_shape[-1])])
    res = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits)
    # if len(output_shape) == 3:
    #     # if our output includes timesteps we need to reshape
    #     return tf.reshape(res, tf.shape(output)[:-1])
    # else:
    #     return res
    return res


INF = 1e6


def logits_from_ter_categorical(state_logit, meta_logit, is_initial_step):
    # scaled_logits = state_logit - tf.reduce_logsumexp(state_logit, axis=1, keepdims=True)
    # bool_is_initial = tf.not_equal(is_initial_step, 0)
    # ter_logit = (1.0 - is_initial_step) * meta_logit + is_initial_step * INF
    ter_logit = meta_logit + state_logit
    full_logits = tf.pad(ter_logit, [[0, 0], [1, 0]])
    return full_logits


# class CategoricalWithProb(ProbType):
#     def __init__(self, num_cat):
#         self.n = num_cat
#
#     def create_dist_vars(self, probs, dtype=tf.float32):
#         old_probs = tf.placeholder(dtype,
#                                    shape=(None, self.n), name="old_probs")
#         old_dist_vars = dict(probs=old_probs)
#         dist_vars = dict(probs=probs)
#         sample = tf.distributions.Categorical(probs=probs).sample()
#         return dist_vars, old_dist_vars, sample, {}
#
#     def log_likelihood_sym(self, x_var, dist_info_vars):
#         one_hot_actions = tf.one_hot(x_var, self.n)
#         logp = -categorical_crossentropy(output=dist_info_vars["probs"], labels=one_hot_actions)
#         return logp
#
#     def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
#         old_p = old_dist_info_vars["probs"]
#         new_p = new_dist_info_vars["probs"]
#         old_dist = tf.distributions.Categorical(probs=old_p)
#         new_dist = tf.distributions.Categorical(probs=new_p)
#         kl = old_dist.kl_divergence(new_dist)
#         return kl
#
#     def entropy(self, dist_info):
#         dist = tf.distributions.Categorical(probs=dist_info["probs"])
#         return dist.entropy()
#
#     def sample(self, dist_info):
#         dist = tf.distributions.Categorical(probs=dist_info["probs"])
#         return dist.sample()
#
#     def kf_loglike(self, action_n, dist_vars, interm_vars):
#         return self.log_likelihood_sym(action_n, dist_vars)


class Categorical(ProbType):
    def __init__(self, num_cat):
        self.n = num_cat

    def create_dist_vars(self, last_layer=None, logits=None, dtype=tf.float32):
        old_logits = tf.placeholder(dtype,
                                    shape=(None, self.n), name="old_logits")
        if logits is None:
            assert last_layer is not None
            logits = tf.layers.dense(last_layer, self.n,
                                     kernel_initializer=tf.variance_scaling_initializer(
                                         scale=1.0, mode="fan_avg", distribution="normal", dtype=dtype))
        old_dist_vars = dict(logits=old_logits)
        dist_vars = dict(logits=logits)
        interm_vars = dict(logits=logits)
        sample = tf.distributions.Categorical(logits=logits).sample()
        return dist_vars, old_dist_vars, sample, interm_vars

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

    def wasserstein_sym(self, old_dist_info_vars, new_dist_info_vars, epsilon=1e-8):
        old_l = old_dist_info_vars["logits"]
        new_l = new_dist_info_vars["logits"]
        old_dist = tf.distributions.Categorical(logits=old_l)
        new_dist = tf.distributions.Categorical(logits=new_l)
        # wasserstein = tf.reduce_sum((old_dist.probs - new_dist.probs)**2/2, axis=-1)
        wasserstein = tf.reduce_sum(tf.abs(old_dist.probs - new_dist.probs) / 2, axis=-1)
        return wasserstein

    def entropy(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.entropy()

    def sample(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.sample()

    def kf_loglike(self, action_n, dist_vars, interm_vars):
        return self.log_likelihood_sym(action_n, dist_vars)

    def reset_exp(self, interm_vars, exploration=0.1):
        return tf.Print(tf.constant(0.0), [], "not implemented")
        # logits = interm_vars["logits"]
        # normed_logits = logits - tf.reduce_logsumexp(logits)
        # final_logits = tf.maximum(normed_logits, np.log(exploration/logits.shape[1].value))
        # return tf.assign(logits, final_logits)


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

        logstd_param = tf.get_variable("logstd", (self.dim,), tf.float32,
                                       tf.zeros_initializer())  # Variance on outputs
        logstd_1a = tf.expand_dims(logstd_param, 0)
        std_1a = tf.exp(logstd_1a)
        logstd_n = tf.tile(logstd_1a, [tf.shape(mean_n)[0], 1])
        std_n = tf.tile(std_1a, [tf.shape(mean_n)[0], 1])
        old_dist_vars = dict(mean=old_mean_n, logstd=old_logstd_n)
        dist_vars = dict(mean=mean_n, logstd=logstd_n)
        interm_vars = dict(std=std_n, logstd_param=logstd_param)
        sample = tf.distributions.Normal(loc=mean_n, scale=std_n).sample()
        return dist_vars, old_dist_vars, sample, interm_vars

    # def gen_dist_info(self, mean, log_std):
    #     return dict(mean=mean, log_std=log_std)

    def gen_exploration_biased_dist_info(self, dist_info_vars, scale=0.0):
        mean = dist_info_vars["mean"]
        mean_no_grad = tf.stop_gradient(mean)
        original_log_stds = dist_info_vars["logstd"]
        grad_scaled_log_stds = scale_positive_gradient_op(original_log_stds, scale=scale)
        return dict(mean=mean_no_grad, logstd=grad_scaled_log_stds)

    def reset_exp(self, interm_vars, std=0.1):
        param = interm_vars["logstd_param"]
        return tf.assign(param, np.log(np.ones(param.get_shape().as_list()) * std))

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, epsilon=1e-8):
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
        denominator = 2 * tf.square(new_std) + epsilon
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

    def wasserstein_sym(self, old_dist_info_vars, new_dist_info_vars, epsilon=1e-8):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["logstd"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["logstd"]
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        wasserstein_terms = tf.square(old_means - new_means) + tf.square(old_std - new_std)

        return tf.reduce_sum(wasserstein_terms, axis=-1)

    def wasserstein_sampled_sym(self, old_dist_info_vars, new_dist_info_vars, logstd_sample_dev=1.0):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["logstd"]
        new_means = new_dist_info_vars["mean"]
        mean_sample = tf.random_normal(tf.shape(new_means), mean=0.0,
                                       stddev=logstd_sample_dev)
        new_log_stds = new_dist_info_vars["logstd"]
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        std_sample = tf.random_normal(tf.shape(new_std), mean=0.0,
                                      stddev=logstd_sample_dev)
        wasserstein_terms = tf.square(new_means + mean_sample - old_means) + tf.square(
            new_std + std_sample - old_std)

        return tf.reduce_sum(wasserstein_terms, axis=-1)

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

    def regulation_loss(self, dist_info):
        log_stds = dist_info["logstd"]
        return tf.reduce_sum(tf.square(tf.maximum(log_stds, 0.0)))

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
