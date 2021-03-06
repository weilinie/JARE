import logging

import tensorflow as tf
from tensorflow.contrib import slim

from libs.ops import (lrelu, fully_connected, conv2d, conv2d_transpose)

logger = logging.getLogger(__name__)


def generator(z, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = slim.fully_connected(z, output_size // 16 * output_size // 16 * f_dim, activation_fn=tf.nn.relu)
    net = tf.reshape(net, [-1, output_size // 16, output_size // 16, f_dim])

    conv2d_trp_argscope = slim.arg_scope(
        [slim.conv2d_transpose], kernel_size=[5, 5], stride=[2, 2], activation_fn=tf.nn.relu
    )

    with conv2d_trp_argscope:
        net = slim.conv2d_transpose(net, f_dim)
        net = slim.conv2d_transpose(net, f_dim)
        net = slim.conv2d_transpose(net, f_dim)
        net = slim.conv2d_transpose(net, c_dim, activation_fn=None)

    out = tf.nn.tanh(net)

    return out


def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = x

    with slim.arg_scope([conv2d], activation_fn=lrelu, is_spectral_norm=True,
                        kernel_size=[5, 5], stride=[2, 2], update_collection=None):
        net = conv2d(net, f_dim, name='conv0')
        net = conv2d(net, f_dim, name='conv1')
        net = conv2d(net, f_dim, name='conv2')
        net = conv2d(net, f_dim, name='conv3')

    net = tf.reshape(net, [-1, output_size // 16 * output_size // 16 * f_dim])
    logits = fully_connected(net, 1, activation_fn=None, is_spectral_norm=True,
                             update_collection=None, name='fc4')
    logits = tf.squeeze(logits, -1)

    return logits
