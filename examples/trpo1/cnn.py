import tensorflow as tf
from tf_utils import lrelu


def cnn_network(input,
                conv_sizes,
                activation_fn=lrelu, initializer_fn=tf.orthogonal_initializer):
    local_scope = "cnn"
    layers = [input]
    with tf.variable_scope(local_scope) as current_scope:
        for (kernel, depth, stride) in conv_sizes:
            # img_features.conv2d(conv_size[0], depth=conv_size[1], activation_fn=lrelu,
            #                     stride=conv_size[2],
            #                     weights=tf.orthogonal_initializer()
            #                     )
            this_layer = tf.layers.conv2d(layers[-1], depth, kernel, stride, activation=activation_fn,
                                          kernel_initializer=initializer_fn())
            layers.append(this_layer)
        net_weights = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope.name)
    return layers, net_weights


class ConvAutoencorder(object):
    def __init__(self, input, conv_sizes,
                 activation_fn=lrelu, initializer_fn=tf.orthogonal_initializer):
        self.encoder_layers, self.encoder_weights = cnn_network(input, conv_sizes, activation_fn, initializer_fn)
        self.decoder_layers = [self.encoder_layers[-1]]
        for (kernel, depth, stride) in reversed(conv_sizes):
            this_layer = tf.layers.conv2d_transpose(self.decoder_layers[-1],
                                                    filters=depth,
                                                    kernel_size=kernel,
                                                    strides=stride,
                                                    activation=activation_fn,
                                                    kernel_initializer=initializer_fn()
                                                    )
            self.decoder_layers.append(this_layer)
        self.output = self.decoder_layers[-1]
        self.reg_loss = tf.reduce_sum(tf.square(input - self.output))
