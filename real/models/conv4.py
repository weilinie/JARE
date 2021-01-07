import logging

import tensorflow as tf
from tensorflow.contrib import slim

from libs.ops import (lrelu, fully_connected, conv2d, conv2d_transpose)

logger = logging.getLogger(__name__)


def generator(z, f_dim, output_size, c_dim, is_training=True):
    bn_kwargs = {
        'decay': 0.9, 'center': True, 'scale': True, 'epsilon': 2e-5,
        'updates_collections': None, 'is_training': is_training
    }

    # Network
    net = slim.fully_connected(z, output_size // 8 * output_size // 8 * 8 * f_dim, activation_fn=tf.nn.relu,
                               normalizer_fn=slim.batch_norm, normalizer_params=None)
    net = tf.reshape(net, [-1, output_size // 8, output_size // 8, 8 * f_dim])

    conv2dtrp_argscope = slim.arg_scope(
        [slim.conv2d_transpose], kernel_size=[4, 4], stride=[2, 2], normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, normalizer_params=None)

    with conv2dtrp_argscope:
        net = slim.conv2d_transpose(net, 4 * f_dim)
        net = slim.conv2d_transpose(net, 2 * f_dim)
        net = slim.conv2d_transpose(net, f_dim)

    net = slim.conv2d(net, c_dim, kernel_size=[3, 3], stride=[1, 1], activation_fn=None,
                      normalizer_fn=slim.batch_norm, normalizer_params=None)

    out = tf.nn.tanh(net)

    return out


def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = x
    with slim.arg_scope([conv2d], activation_fn=lrelu, is_spectral_norm=False,
                        update_collection=None):
        net = conv2d(net, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv0_0')
        net = conv2d(net, f_dim, kernel_size=[4, 4], stride=[2, 2], name='conv0_1')
        net = conv2d(net, 2 * f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv1_0')
        net = conv2d(net, 2 * f_dim, kernel_size=[4, 4], stride=[2, 2], name='conv1_1')
        net = conv2d(net, 4 * f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv2_0')
        net = conv2d(net, 4 * f_dim, kernel_size=[4, 4], stride=[2, 2], name='conv2_1')
        net = conv2d(net, 8 * f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv3_0')

    net = tf.reshape(net, [-1, output_size // 8 * output_size // 8 * 8 * f_dim])
    logits = fully_connected(net, 1, activation_fn=None, is_spectral_norm=False,
                             update_collection=None)
    logits = tf.squeeze(logits, -1)

    return logits

