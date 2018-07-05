import tensorflow as tf
import numpy as np
from tf_utils import scale_positive_gradient_op
import logging
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


def gaussian_loglikelihood(x_var, means, log_stds=None, stds=None):
    if log_stds is not None:
        zs = (x_var - means) * tf.exp(-log_stds)
        log_det_std = tf.reduce_sum(log_stds, -1)
        log_pi = - log_det_std - \
                 0.5 * tf.reduce_sum(tf.square(zs), -1) - \
           0.5 * means.get_shape()[-1].value * np.log(2 * np.pi)
    else:
        log_pi = - tf.reduce_sum(tf.log(stds), axis=1) - 0.5 * tf.log(
            2.0 * np.pi) * means.get_shape()[-1].value - 0.5 * tf.reduce_sum(
            tf.square(means - x_var) / (tf.square(stds)),
            axis=1)
    return log_pi


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
                                         scale=0.01, mode="fan_avg", distribution="normal", dtype=dtype))
        old_dist_vars = dict(logits=old_logits)
        dist_vars = dict(logits=logits)
        interm_vars = dict(logits=logits)
        sample = tf.distributions.Categorical(logits=logits).sample()
        return dist_vars, old_dist_vars, sample, interm_vars

    def log_likelihood_sym(self, x_var, dist_info_vars, interim_vars=None):
        one_hot_actions = tf.one_hot(x_var, self.n)
        logp = -tf.nn.softmax_cross_entropy_with_logits(logits=dist_info_vars["logits"], labels=one_hot_actions)
        return logp

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, interim_vars=None, action_n=None):
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

    def regulation_loss(self, dist_info, min_prob=0.001):
        logits = dist_info["logits"]
        logits_logsumexp = tf.reduce_logsumexp(logits, axis=-1)
        min_logit = tf.reduce_min(logits, axis=-1)
        min_logprob = min_logit - logits_logsumexp
        reg_loss = tf.square(tf.maximum(np.log(min_prob) - min_logprob, 0.))
        return reg_loss

    def entropy(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.entropy()

    def sample(self, dist_info):
        dist = tf.distributions.Categorical(logits=dist_info["logits"])
        return dist.sample()

    # def kf_loglike(self, action_n, dist_vars, interm_vars):
    #     return self.log_likelihood_sym(action_n, dist_vars)

    def reset_exp(self, interm_vars, exploration=0.1):
        return tf.Print(tf.constant(0.0), [], "not implemented")
        # logits = interm_vars["logits"]
        # normed_logits = logits - tf.reduce_logsumexp(logits)
        # final_logits = tf.maximum(normed_logits, np.log(exploration/logits.shape[1].value))
        # return tf.assign(logits, final_logits)


KEY_MEAN = "mean"
KEY_LOGSTD = "logstd"

class DiagonalGaussian(ProbType):
    def __init__(self, dim, normalize_wass=False):
        logging.debug("gaussian params:{}".format(locals()))
        self._dim = dim
        self.normalize_wass = normalize_wass

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

    def gen_exploration_biased_dist_info(self, dist_info_vars):
        mean = dist_info_vars[KEY_MEAN]
        mean_no_grad = tf.stop_gradient(mean)
        original_log_std = dist_info_vars[KEY_LOGSTD]
        # logstd_value = tf.stop_gradient(dist_info_vars[KEY_LOGSTD])
        # logstd_with_lower_bound = tf.maximum(original_log_std, logstd_value)
        positive_grad_logstd = scale_positive_gradient_op(original_log_std)
        return dict(mean=mean_no_grad, logstd=positive_grad_logstd)

    def fixed_std_dist_info(self, dist_info_vars):
        mean = dist_info_vars[KEY_MEAN]
        original_log_stds = dist_info_vars[KEY_LOGSTD]
        logstd_no_grad = tf.stop_gradient(original_log_stds)
        return dict(mean=mean, logstd=logstd_no_grad)

    def fixed_mean_dist_info(self, dist_info_vars):
        mean = dist_info_vars[KEY_MEAN]
        fixed_mean = tf.stop_gradient(mean)
        logstd = dist_info_vars[KEY_LOGSTD]
        return dict(mean=fixed_mean, logstd=logstd)

    def reset_exp(self, interm_vars, std=0.1):
        param = interm_vars["logstd_param"]
        return tf.assign(param, np.log(np.ones(param.get_shape().as_list()) * std))

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars,
               action_n=None, interim_vars=None, epsilon=1e-8):
        old_means = old_dist_info_vars[KEY_MEAN]
        old_log_stds = old_dist_info_vars[KEY_LOGSTD]
        new_means = new_dist_info_vars[KEY_MEAN]
        new_log_stds = new_dist_info_vars[KEY_LOGSTD]
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

    def log_likelihood_sym(self, x_var, dist_info_vars, interim_vars=None):
        """
        \frac{1}{(2\pi)^{\frac{n}{2}}\sigma_\theta}exp(-(\frac{a-\mu_{\pi_\theta}}{2\sigma_\theta})^2)
        :param x_var:
        :param dist_info_vars:
        :return:
        """
        if interim_vars is None:
            means = dist_info_vars[KEY_MEAN]
            log_stds = dist_info_vars[KEY_LOGSTD]
            return gaussian_loglikelihood(x_var, means, log_stds=log_stds)
        else:
            means = dist_info_vars[KEY_MEAN]
            stds = interim_vars["std"]
            return gaussian_loglikelihood(x_var, means, stds=stds)

    def norm_wass(self, wass, old_std, new_std):
        return wass / (0.5 * old_std ** 2 + 0.5 * tf.stop_gradient(new_std) ** 2)

    def wasserstein_sym(self, old_dist_info_vars, new_dist_info_vars, action_n=None, epsilon=1e-8):
        old_means = old_dist_info_vars[KEY_MEAN]
        old_log_stds = old_dist_info_vars[KEY_LOGSTD]
        new_means = new_dist_info_vars[KEY_MEAN]
        new_log_stds = new_dist_info_vars[KEY_LOGSTD]
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        wasserstein_terms = tf.square(old_means - new_means) + tf.square(old_std - new_std)
        if self.normalize_wass:
            logging.debug("using normalized wass")
            wasserstein_terms = self.norm_wass(wasserstein_terms, old_std, new_std)

        return tf.reduce_sum(wasserstein_terms, axis=-1)

    def wasserstein_sampled_sym(self, old_dist_info_vars,
                                new_dist_info_vars, interim_vars, action_n=None,
                                logstd_sample_dev=1.0):
        old_means = old_dist_info_vars[KEY_MEAN]
        old_log_stds = old_dist_info_vars[KEY_LOGSTD]
        new_means = new_dist_info_vars[KEY_MEAN]
        mean_sample = tf.random_normal(tf.shape(new_means), mean=0.0,
                                       stddev=logstd_sample_dev)
        new_log_stds = new_dist_info_vars[KEY_LOGSTD]
        old_std = tf.exp(old_log_stds)
        new_std = interim_vars["std"]
        std_sample = tf.random_normal(tf.shape(new_std), mean=0.0,
                                      stddev=logstd_sample_dev)
        wasserstein_terms = tf.square(new_means + mean_sample - old_means) + tf.square(
            new_std + std_sample - old_std)
        if self.normalize_wass:
            logging.debug("using normalized wass")
            wasserstein_terms = self.norm_wass(wasserstein_terms, old_std, new_std)

        return tf.reduce_sum(wasserstein_terms, axis=-1)

    def sample(self, dist_info):
        means = dist_info[KEY_MEAN]
        log_stds = dist_info[KEY_LOGSTD]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info[KEY_MEAN]
        log_stds = dist_info[KEY_LOGSTD]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info[KEY_LOGSTD]
        return tf.reduce_sum(log_stds, 1) + 0.5 * np.log(2 * np.pi * np.e) * self.dim

    def regulation_loss(self, dist_info):
        log_stds = dist_info[KEY_LOGSTD]
        return tf.reduce_sum(tf.square(tf.maximum(log_stds, 0.0)))

    # def log_likelihood_from_std(self, action_n, mean_n, std_n):
    #     logprob_n = - tf.reduce_sum(tf.log(std_n), axis=1) - 0.5 * tf.log(
    #         2.0 * np.pi) * self.dim - 0.5 * tf.reduce_sum(
    #         tf.square(mean_n - action_n) / (tf.square(std_n)),
    #         axis=1)
    #     return logprob_n
    #
    # def kf_loglike(self, action_n, dist_vars, interm_vars):
    #     mean_n = dist_vars[KEY_MEAN]
    #     std_n = interm_vars["std"]
    #     logprob_n = self.log_likelihood_from_std(action_n, mean_n, std_n)
    #     return logprob_n

    @property
    def dist_info_keys(self):
        return [KEY_MEAN, KEY_LOGSTD]


class RobustMixtureGaussian(ProbType):
    def __init__(self, dim,
                 exploration_prob=0.05, max_std=1.0, std_ratio=4., clip=False, normalize_wass=False):
        logging.debug("mixture params:{}".format(locals()))
        self.dim = dim
        self.exploration_prob = exploration_prob
        self.max_std = max_std
        self.std_ratio = std_ratio
        self.clip = clip
        self.normalize_wass = normalize_wass

    @property
    def main_prob(self):
        return 1 - self.exploration_prob

    def get_stdn_from_logstd_param(self, logstd_param, nsample):
        logstd_1a = tf.expand_dims(logstd_param, 0)
        std_1a = tf.exp(logstd_1a)
        logstd_n = tf.tile(logstd_1a, [nsample, 1])
        std_n = tf.tile(std_1a, [nsample, 1])
        return logstd_n, std_n

    def get_distrobust_logstd_from_main_logstd(self, logstd):
        distrobust_logstd = logstd + np.log(self.std_ratio)
        if self.clip:
            distrobust_logstd = tf.minimum(distrobust_logstd, np.log(self.max_std))
        return distrobust_logstd

    def get_distrobust_std_from_main_std(self, std):
        distrobust_std = std * float(self.std_ratio)
        if self.clip:
            distrobust_std = tf.minimum(distrobust_std, self.max_std)
        return distrobust_std

    def _get_sample(self, mean_n, std_n):
        sample_main = tf.distributions.Normal(loc=mean_n, scale=std_n).sample()
        distrobust_stdn = self.get_distrobust_std_from_main_std(std_n)
        sample_distrobust = tf.distributions.Normal(loc=mean_n, scale=distrobust_stdn).sample()
        component_sample = tf.cast(
            tf.distributions.Categorical(probs=[self.main_prob, self.exploration_prob]).sample(tf.shape(sample_main)),
            tf.float32)
        sample = (1. - component_sample) * sample_main + component_sample * sample_distrobust
        return sample

    def create_dist_vars(self, last_layer, dtype=tf.float32):
        old_mean_n = tf.placeholder(dtype, shape=(None, self.dim),
                                    name="oldaction_dist_means")
        old_logstd_n = tf.placeholder(dtype, shape=(None, self.dim),
                                      name="oldaction_dist_logstds")
        mean_n = tf.layers.dense(last_layer, self.dim,  # activation=tf.tanh,
                                 kernel_initializer=tf.variance_scaling_initializer(
                                     scale=1.0, mode="fan_avg", distribution="normal", dtype=dtype))
        nsample = tf.shape(mean_n)[0]

        logstd_param = tf.get_variable("logstd", (self.dim,), tf.float32,
                                       tf.constant_initializer(
                                           value=np.log(0.25)))  # Variance on outputs
        logstd_n, std_n = self.get_stdn_from_logstd_param(logstd_param, nsample)
        old_dist_vars = dict(mean=old_mean_n, logstd=old_logstd_n)
        dist_vars = dict(mean=mean_n, logstd=logstd_n)
        interm_vars = dict(std=std_n, logstd_param=logstd_param)
        sample = self._get_sample(mean_n, std_n)
        return dist_vars, old_dist_vars, sample, interm_vars

    def _distrobust_info_vars_from_main(self, distmain_info_vars, interim_vars=None):

        distrobust_info_vars = {KEY_MEAN: distmain_info_vars[KEY_MEAN]}
        distrobust_info_vars[KEY_LOGSTD] = \
            self.get_distrobust_logstd_from_main_logstd(distmain_info_vars[KEY_LOGSTD])
        if interim_vars is None:
            return distrobust_info_vars
        else:
            interim_vars = {"std": self.get_distrobust_std_from_main_std(interim_vars["std"])}
            return distrobust_info_vars, interim_vars

    def log_likelihood_sym(self, x_var, dist_info_vars, interim_vars=None):
        dummy_gaussian = DiagonalGaussian(self.dim)
        main_log_likelihood = dummy_gaussian.log_likelihood_sym(x_var, dist_info_vars, interim_vars=interim_vars)
        distrobust_info_vars = self._distrobust_info_vars_from_main(dist_info_vars)
        if interim_vars is not None:
            distrobust_interim_vars = {"std": self.get_distrobust_std_from_main_std(interim_vars["std"])}
        else:
            distrobust_interim_vars = None
        distrobust_log_likelihood = dummy_gaussian.log_likelihood_sym(
            x_var, distrobust_info_vars, interim_vars=distrobust_interim_vars)
        main_logp = main_log_likelihood + np.log(self.main_prob)
        distrobust_logp = distrobust_log_likelihood + np.log(self.exploration_prob)
        offset = tf.maximum(main_logp, distrobust_logp)
        final_log_likelihood = offset + tf.log(tf.exp(main_logp - offset)
                                               + tf.exp(distrobust_logp - offset))
        return final_log_likelihood

    def norm_wass(self, wass, old_std, new_std):
        return wass / (0.5 * old_std ** 2 + 0.5 * tf.stop_gradient(new_std) ** 2)

    def wasserstein_sym(self, old_dist_info_vars, new_dist_info_vars, action_n=None):
        dummy_gaussian = DiagonalGaussian(self.dim)
        old_distrobust_info_vars = self._distrobust_info_vars_from_main(old_dist_info_vars)
        new_distrobust_info_vars = self._distrobust_info_vars_from_main(new_dist_info_vars)
        main_wass = dummy_gaussian.wasserstein_sym(old_dist_info_vars, new_dist_info_vars)
        distroobust_wass = dummy_gaussian.wasserstein_sym(old_distrobust_info_vars, new_distrobust_info_vars)
        final_wass = self.main_prob * main_wass + self.exploration_prob * distroobust_wass

        if self.normalize_wass:
            logging.debug("using normalized wass")
            final_wass = self.norm_wass(final_wass,
                                        tf.exp(old_dist_info_vars[KEY_LOGSTD]),
                                        tf.exp(new_dist_info_vars[KEY_LOGSTD]))
        return final_wass

    def wasserstein_sampled_sym(self, old_dist_info_vars, new_dist_info_vars,
                                interim_vars, logstd_sample_dev=1.0):
        dummy_gaussian = DiagonalGaussian(self.dim)
        old_distrobust_info_vars = self._distrobust_info_vars_from_main(old_dist_info_vars)
        new_distrobust_info_vars = self._distrobust_info_vars_from_main(new_dist_info_vars)
        main_wass = dummy_gaussian.wasserstein_sampled_sym(
            old_dist_info_vars, new_dist_info_vars,
            interim_vars, logstd_sample_dev)
        distroobust_wass = dummy_gaussian.wasserstein_sampled_sym(
            old_distrobust_info_vars, new_distrobust_info_vars,
            {"std": self.get_distrobust_std_from_main_std(interim_vars["std"])},
            logstd_sample_dev)
        final_wass = self.main_prob * main_wass + self.exploration_prob * distroobust_wass
        if self.normalize_wass:
            logging.debug("using normalized wass")
            final_wass = self.norm_wass(final_wass,
                                        tf.exp(old_dist_info_vars[KEY_LOGSTD]),
                                        tf.exp(new_dist_info_vars[KEY_LOGSTD]))
        return final_wass

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, action_n=None, interim_vars=None):
        old_log_pi = self.log_likelihood_sym(action_n, old_dist_info_vars, interim_vars=None)
        new_log_pi = self.log_likelihood_sym(
            action_n,  # self._get_sample(mean_n=new_dist_info_vars[KEY_MEAN], std_n=interim_vars["std"]),
            new_dist_info_vars,
            interim_vars
        )
        kl_sampled = old_log_pi - new_log_pi
        return kl_sampled

    def entropy(self, dist_info):
        log_stds = dist_info[KEY_LOGSTD]
        return tf.reduce_sum(log_stds, 1) + 0.5 * np.log(2 * np.pi * np.e) * self.dim

    def reset_exp(self, interm_vars, std=0.1):
        param = interm_vars["logstd_param"]
        return tf.assign(param, np.log(np.ones(param.get_shape().as_list()) * std))

    def regulation_loss(self, dist_info):
        return 0.
