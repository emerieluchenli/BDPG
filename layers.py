import numpy as np
import tensorflow as tf
from math import ceil


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name) as scope:
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            conv_o_b = tf.nn.bias_add(conv, bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(scope, conv_o_dr)

    return conv_o


def conv2d_transpose(name, x, w=None, output_shape=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1), l2_strength=0.0,
                     bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.shape[-1]]
        if w == None:
            w = get_deconv_filter(kernel_shape, l2_strength)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)
        if isinstance(bias, float):
            bias = tf.get_variable('layer_biases', [output_shape[-1]], initializer=tf.constant_initializer(bias))
        conv_o_b = tf.nn.bias_add(deconv, bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr

    return conv_o


def fully_connected(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(),
                    l2_strength=0.0, bias=0.0,  activation=None, batchnorm_enabled=False,
                    dropout_keep_prob=-1, is_training=True):
    n_in = x.get_shape()[-1].value

    with tf.variable_scope(name):
        if w == None:
            w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        dense_o_b = tf.nn.bias_add(tf.matmul(x, w), bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.nn.dropout(dense_a, dropout_keep_prob)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


# def flatten(x):
#     all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
#     o = tf.reshape(x, [-1, all_dims_exc_first])
#     return o


def flatten(x):
    return tf.contrib.layers.flatten(x)


def max_pool_2d(x, size=(2, 2)):
    size_x, size_y = size
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, size_x, size_y, 1], padding='VALID',
                          name='pooling')


def upsample_2d(x, size=(2, 2)):
    h, w, _ = x.get_shape().as_list()[1:]
    size_x, size_y = size
    output_h = h * size_x
    output_w = w * size_y
    return tf.image.resize_bilinear(x, (output_h, output_w), align_corners=None, name='upsampling')


def variable_with_weight_decay(kernel_shape, initializer, wd):
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w


def get_deconv_filter(f_shape, l2_strength):
    # initializer for bilinear convolution transpose filters
    width = f_shape[0]
    height = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return variable_with_weight_decay(weights.shape, init, l2_strength)


def noise_and_argmax(logits, amp=1.):
    # add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)) * amp, 1)


def a3c_entropy(logits):
    # entropy proposed in A3C
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def softmax_entropy(p0):
    # information theory entropy by Shannon
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)


def mse(y, y_pred):
    return tf.losses.mean_squared_error(y, y_pred)


def orthogonal_initializer(scale=1.):
    # Orthogonal Initializer that uses SVD. The unused variables are just for passing in tf
    # shape = tuple(shape)
    # if len(shape) == 2:
    #     flat_shape = shape
    # elif len(shape) == 4:  # assumes NHWC
    #     flat_shape = (np.prod(shape[:-1]), shape[-1])
    # else:
    #     raise NotImplementedError
    # a = np.random.normal(0.0, 1.0, flat_shape)
    # u, _, v = np.linalg.svd(a, full_matrices=False)
    # q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    # q = q.reshape(shape)
    # return float(scale * q[:shape[0], :shape[1]])

    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init




