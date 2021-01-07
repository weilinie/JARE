import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers, functools
from tensorflow.contrib import slim
from tensorflow.python.training import moving_averages

from libs.sn import spectral_norm


def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak * x)


def get_batch_moments(x, is_training=True, decay=0.99, is_init=False):
    x_shape = x.get_shape()
    axis = list(range(len(x_shape) - 1))

    moving_mean = tf.get_variable('moving_mean', x_shape[-1:], tf.float32, trainable=False,
                                  initializer=tf.zeros_initializer())
    moving_variance = tf.get_variable('moving_var', x_shape[-1:], tf.float32, trainable=False,
                                      initializer=tf.ones_initializer())

    if is_init:
        mean, variance = tf.nn.moments(x, axis)
    elif is_training:
        # Calculate the moments based on the individual batch.
        mean, variance = tf.nn.moments(x, axis, shift=moving_mean)
        # Update the moving_mean and moving_variance moments.
        update_moving_mean = moving_mean.assign_sub((1 - decay) * (moving_mean - mean))
        update_moving_variance = moving_variance.assign_sub(
            (1 - decay) * (moving_variance - variance))
        # Make sure the updates are computed here.
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            mean, variance = tf.identity(mean), tf.identity(variance)
    else:
        mean, variance = moving_mean, moving_variance
    return mean, tf.sqrt(variance + 1e-8)


def get_input_moments(x, is_init=False, name=None):
    '''Input normalization'''
    with tf.variable_scope(name, default_name='input_norm'):
        if is_init:
            # data based initialization of parameters
            mean, variance = tf.nn.moments(x, [0])
            std = tf.sqrt(variance + 1e-8)
            mean0 = tf.get_variable('mean0', dtype=tf.float32,
                                    initializer=mean, trainable=False)
            std0 = tf.get_variable('std0', dtype=tf.float32,
                                   initializer=std, trainable=False)
            return mean, std

        else:
            mean0 = tf.get_variable('mean0')
            std0 = tf.get_variable('std0')
            tf.assert_variables_initialized([mean0, std0])
            return mean0, std0


@add_arg_scope
def fully_connected(inputs, output_dim, activation_fn=None, is_spectral_norm=False, update_collection=None,
                    is_xavier_init=True, with_biases=True, with_w=False, name="fc"):
    ''' fully connected layer '''
    in_shape = inputs.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        if is_xavier_init:
            weight = tf.get_variable("w", [in_shape[1], output_dim], dtype=tf.float32,
                                     initializer=initializers.xavier_initializer())
        else:
            weight = tf.get_variable("w", [in_shape[1], output_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
        if is_spectral_norm:
            out = tf.matmul(inputs, spectral_norm(weight, update_collection=update_collection))
        else:
            out = tf.matmul(inputs, weight)
        if with_biases:
            bias = tf.get_variable("b", [output_dim],
                                   initializer=tf.constant_initializer(0.0))
            out = tf.reshape(tf.nn.bias_add(out, bias), out.get_shape())
        if activation_fn is not None:
            out = activation_fn(out)
        if with_w:
            return out, weight
        else:
            return out


@add_arg_scope
def conv2d(inputs, output_dim, kernel_size=[3, 3], stride=[1, 1], activation_fn=None, padding="SAME",
           is_xavier_init=True, is_spectral_norm=False, update_collection=None, with_biases=True,
           with_w=False, name="conv2d"):
    '''convolutional layer'''
    in_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        # filter : [height, width, in_channels, output_channels]
        if is_xavier_init:
            w = tf.get_variable("w", kernel_size + [in_shape[-1], output_dim],
                                initializer=initializers.xavier_initializer())
        else:
            w = tf.get_variable("w", kernel_size + [in_shape[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        if is_spectral_norm:
            conv = tf.nn.conv2d(inputs, spectral_norm(w, update_collection=update_collection),
                                strides=[1] + stride + [1], padding=padding)
        else:
            conv = tf.nn.conv2d(inputs, w, strides=[1] + stride + [1], padding=padding)
        if with_biases:
            biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if activation_fn is not None:
            conv = activation_fn(conv)

        if with_w:
            return conv, w
        else:
            return conv


@add_arg_scope
def conv2d_transpose(inputs, output_dim, kernel_size=[3, 3], stride=[1, 1], activation_fn=None,
                     padding="SAME", is_xavier_init=True, is_spectral_norm=False, update_collection=None,
                     with_biases=True, with_w=False, name="conv2d_transpose"):
    '''transposed convolutional layer'''
    in_shape = inputs.get_shape().as_list()
    if padding == "SAME":
        output_shape = [in_shape[0], in_shape[1] * stride[0],
                        in_shape[2] * stride[1], output_dim]
    else:
        output_shape = [in_shape[0], in_shape[1] * stride[0] + kernel_size[0] -
                        1, in_shape[2] * stride[1] + kernel_size[1] - 1, output_dim]
    with tf.variable_scope(name) as scope:
        # filter : [height, width, output_channels, in_channels]
        if is_xavier_init:
            w = tf.get_variable("w", kernel_size + [output_dim, in_shape[-1]],
                                initializer=initializers.xavier_initializer())
        else:
            w = tf.get_variable("w", kernel_size + [output_dim, in_shape[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        if is_spectral_norm:
            deconv = tf.nn.conv2d_transpose(inputs, spectral_norm(w, update_collection=update_collection),
                                            output_shape=output_shape,
                                            strides=[1] + stride + [1], padding=padding)
        else:
            deconv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                            strides=[1] + stride + [1], padding=padding)
        if with_biases:
            biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if activation_fn is not None:
            deconv = activation_fn(deconv)
        if with_w:
            return deconv, w
        else:
            return deconv


def upsample(x):
    xshape = [int(t) for t in x.get_shape()]
    # ipdb.set_trace()
    x_rs = tf.reshape(x, [xshape[0] * xshape[1], 1, xshape[2] * xshape[3]])
    x_rs = tf.tile(x_rs, [1, 2, 1])
    x_rs = tf.reshape(x_rs, [xshape[0] * 2 * xshape[1] * xshape[2], 1, xshape[3]])
    x_rs = tf.tile(x_rs, [1, 2, 1])
    x_out = tf.reshape(x_rs, [xshape[0], 2 * xshape[1], 2 * xshape[2], xshape[3]])

    return x_out


def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def int_shape(x):
    return list(map(int, x.get_shape()))


def train_opt(opt_type, lr, beta1=0.9, beta2=0.999):
    if opt_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr)
    elif opt_type == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    else:
        raise Exception("Unknown opt_type")


def batch_norm(input, is_training=True, momentum=0.9, epsilon=2e-5, in_place_update=True, name="batch_norm"):
    if in_place_update:
        return tf.contrib.layers.batch_norm(input,
                                            decay=momentum,
                                            center=True,
                                            scale=True,
                                            epsilon=epsilon,
                                            updates_collections=None,
                                            is_training=is_training,
                                            scope=name)
    else:
        return tf.contrib.layers.batch_norm(input,
                                            decay=momentum,
                                            center=True,
                                            scale=True,
                                            epsilon=epsilon,
                                            is_training=is_training,
                                            scope=name)


def ConvMeanPool(inputs, output_dim, kernel_size, name, is_spectral_norm, update_collection, biases=True):
    output = inputs
    output = conv2d(output, output_dim, kernel_size, name=name, is_spectral_norm=is_spectral_norm,
                    update_collection=update_collection, with_biases=biases)
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    return output


def MeanPoolConv(inputs, output_dim, kernel_size, name, is_spectral_norm, update_collection, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, ::2, ::2, :], output[:, 1::2, ::2, :], output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    output = conv2d(output, output_dim, kernel_size, name=name, is_spectral_norm=is_spectral_norm,
                    update_collection=update_collection, with_biases=biases)
    return output


def UpsampleConv(inputs, output_dim, kernel_size, name, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=-1)
    output = tf.depth_to_space(output, 2)
    output = conv2d(output, output_dim, kernel_size, name=name, with_biases=biases)
    return output


def genResBlock(inputs, output_dim, kernel_size, name, resample=None, is_spectral_norm=False, update_collection=None,
                activation_fn=tf.nn.relu, is_training=True, with_biases=True):
    """
    resample: None, 'down', or 'up'
    """
    input_dim = inputs.get_shape().as_list()[-1]
    if resample == 'down':
        conv_1 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)  # TODO: here is it output_dim=input_dim?
        conv_2 = functools.partial(ConvMeanPool, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
        conv_shortcut = functools.partial(ConvMeanPool, is_spectral_norm=is_spectral_norm,
                                          update_collection=update_collection)
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(conv2d, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = functools.partial(conv2d, is_spectral_norm=is_spectral_norm,
                                          update_collection=update_collection)
        conv_1 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
        conv_2 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs, output_dim, kernel_size=[1, 1], name=name + '.Shortcut', biases=with_biases)

    output = inputs
    output = batch_norm(output, is_training=is_training, name=name + '.N1')
    output = activation_fn(output)
    output = conv_1(output, kernel_size=kernel_size, name=name + '.Conv1')
    output = batch_norm(output, is_training=is_training, name=name + '.N2')
    output = activation_fn(output)
    output = conv_2(output, kernel_size=kernel_size, name=name + '.Conv2')

    return shortcut + output


def disResBlock(inputs, output_dim, kernel_size, name, resample=None, is_spectral_norm=False, update_collection=None,
                activation_fn=tf.nn.relu, is_training=True, with_biases=True):
    """
    resample: None, 'down', or 'up'
    """
    input_dim = inputs.get_shape().as_list()[-1]
    if resample == 'down':
        conv_1 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)  # TODO: here is it output_dim=input_dim?
        conv_2 = functools.partial(ConvMeanPool, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
        conv_shortcut = functools.partial(ConvMeanPool, is_spectral_norm=is_spectral_norm,
                                          update_collection=update_collection)
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(conv2d, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = functools.partial(conv2d, is_spectral_norm=is_spectral_norm,
                                          update_collection=update_collection)
        conv_1 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
        conv_2 = functools.partial(conv2d, output_dim=output_dim, is_spectral_norm=is_spectral_norm,
                                   update_collection=update_collection)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs, output_dim, kernel_size=[1, 1], name=name + '.Shortcut', biases=with_biases)

    output = inputs
    output = batch_norm(output, is_training=is_training, name=name + '.N1')
    output = activation_fn(output)
    output = conv_1(output, kernel_size=kernel_size, name=name + '.Conv1')
    output = batch_norm(output, is_training=is_training, name=name + '.N2')
    output = activation_fn(output)
    output = conv_2(output, kernel_size=kernel_size, name=name + '.Conv2')

    return shortcut + output
