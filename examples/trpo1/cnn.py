import tensorflow as tf
from tf_utils import lrelu


def cnn_network(input,
                conv_sizes,
                activation_fn=lrelu, initializer_fn=tf.orthogonal_initializer,
                fc_sizes=(0,), fc_activation=tf.tanh,
                trainable=True):
    cnn_scope = "cnn"
    layers = [input]
    with tf.variable_scope(cnn_scope) as current_scope:
        for (kernel, depth, stride) in conv_sizes:
            # img_features.conv2d(conv_size[0], depth=conv_size[1], activation_fn=lrelu,
            #                     stride=conv_size[2],
            #                     weights=tf.orthogonal_initializer()
            #                     )
            this_layer = tf.layers.conv2d(layers[-1], depth, kernel, stride, activation=activation_fn,
                                          kernel_initializer=initializer_fn(),
                                          padding="same",
                                          trainable=trainable)
            layers.append(this_layer)
        net_weights = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope.name)
    fc_scope = "fc"
    if fc_sizes[0] > 0:
        with tf.variable_scope(fc_scope) as current_scope:
            final_cnn_flat = tf.layers.flatten(layers[-1])
            current_layer = final_cnn_flat
            for num_hid in fc_sizes:
                current_layer = tf.layers.dense(current_layer, num_hid, activation=fc_activation,
                                                kernel_initializer=initializer_fn())
                layers.append(current_layer)
            fc_weights = tf.trainable_variables(current_scope.name)
    else:
        fc_weights = []
    return layers, net_weights, fc_weights


class ConvAutoencorder(object):
    def __init__(self, input, conv_sizes,
                 activation_fn=lrelu, initializer_fn=tf.orthogonal_initializer,
                 num_fc=(0,), fc_activation=tf.tanh):
        self.encoder_layers, self.encoder_weights, self.fc_weights = cnn_network(input, conv_sizes, activation_fn,
                                                                                 initializer_fn,
                                                                                 fc_sizes=num_fc,
                                                                                 fc_activation=fc_activation)
        cnn_final_layer_num = len(conv_sizes)
        self.cnn_layers = self.encoder_layers[0:(cnn_final_layer_num + 1)]
        self.decoder_layers = [self.cnn_layers[-1]]
        local_scope = "dec_cnn"
        with tf.variable_scope(local_scope) as current_scope:
            for (i, (kernel, depth, stride)) in enumerate(reversed(conv_sizes)):
                this_layer = tf.layers.conv2d_transpose(self.decoder_layers[-1],
                                                        filters=self.cnn_layers[-2 - i].shape[3],
                                                        kernel_size=kernel,
                                                        strides=stride,
                                                        activation=activation_fn,
                                                        kernel_initializer=initializer_fn(),
                                                        padding="same"
                                                        )
                self.decoder_layers.append(this_layer)
            self.decoder_weights = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope.name)
        self.output = self.decoder_layers[-1]
        self.reg_loss = tf.reduce_mean(tf.square(input - self.output))
        self.total_var_list = self.encoder_weights + self.decoder_weights + self.fc_weights


class ConvFcAutoencorder(object):
    def __init__(self, input, conv_sizes,
                 activation_fn=lrelu, initializer_fn=tf.orthogonal_initializer,
                 num_fc=(0,), fc_activation=tf.tanh):
        self.encoder_layers, self.en_cnn_weights, self.en_fc_weights = cnn_network(input, conv_sizes, activation_fn,
                                                                                   initializer_fn,
                                                                                   fc_sizes=num_fc,
                                                                                   fc_activation=fc_activation)
        cnn_final_layer_num = len(conv_sizes)
        self.cnn_layers = self.encoder_layers[0:(cnn_final_layer_num + 1)]
        self.decoder_layers = [self.cnn_layers[-1]]
        local_scope = "dec_cnn"
        with tf.variable_scope(local_scope) as current_scope:
            for (i, (kernel, depth, stride)) in enumerate(reversed(conv_sizes)):
                this_layer = tf.layers.conv2d_transpose(self.decoder_layers[-1],
                                                        filters=self.cnn_layers[-2 - i].shape[3],
                                                        kernel_size=kernel,
                                                        strides=stride,
                                                        activation=activation_fn,
                                                        kernel_initializer=initializer_fn(),
                                                        padding="same"
                                                        )
                self.decoder_layers.append(this_layer)
            self.decoder_weights = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope.name)
        self.output = self.decoder_layers[-1]
        self.reg_loss = tf.reduce_mean(tf.square(input - self.output))
        self.total_var_list = self.encoder_weights + self.decoder_weights + self.fc_weights