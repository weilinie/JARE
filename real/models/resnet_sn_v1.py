import logging

import tensorflow as tf
from tensorflow.contrib import slim

from libs.ops import (lrelu, fully_connected, conv2d, conv2d_transpose)

logger = logging.getLogger(__name__)


def generator(z, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = slim.fully_connected(z, output_size // 16 * output_size // 16 * f_dim, activation_fn=tf.nn.relu)
    net = tf.reshape(net, [-1, output_size // 16, output_size // 16, f_dim])

    argscope_conv2d_trp = slim.arg_scope(
        [slim.conv2d_transpose], kernel_size=[4, 4], stride=[2, 2], activation_fn=tf.nn.relu)
    argscope_conv2d = slim.arg_scope(
        [slim.conv2d], kernel_size=[3, 3], stride=[1, 1], activation_fn=tf.nn.relu)

    with argscope_conv2d, argscope_conv2d_trp:
        net = slim.conv2d_transpose(net, f_dim)
        dnet = slim.conv2d(net, f_dim // 2)
        net += 1e-1 * slim.conv2d(dnet, f_dim)

        net = slim.conv2d_transpose(net, f_dim)
        dnet = slim.conv2d(net, f_dim // 2)
        net += 1e-1 * slim.conv2d(dnet, f_dim)

        net = slim.conv2d_transpose(net, f_dim)
        dnet = slim.conv2d(net, f_dim // 2)
        net += 1e-1 * slim.conv2d(dnet, f_dim)

        net = slim.conv2d_transpose(net, c_dim, activation_fn=None)

    out = tf.nn.tanh(net)

    return out


def discriminator(x, f_dim, output_size, c_dim, is_training=True):
    # Network
    net = x

    argscope_conv2d = slim.arg_scope([conv2d], kernel_size=[4, 4], stride=[2, 2],
                                     activation_fn=tf.nn.relu, is_spectral_norm=True, update_collection=None)
    with argscope_conv2d:
        net = conv2d(net, f_dim, name='conv0_0')
        dnet = conv2d(net, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv0_1')
        net += 1e-1 * conv2d(dnet, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv0_2')

        net = conv2d(net, f_dim, name='conv1_0')
        dnet = conv2d(net, f_dim // 2, kernel_size=[3, 3], stride=[1, 1], name='conv1_1')
        net += 1e-1 * conv2d(dnet, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv1_2')

        net = conv2d(net, f_dim, name='conv2_0')
        dnet = conv2d(net, f_dim // 2, kernel_size=[3, 3], stride=[1, 1], name='conv2_1')
        net += 1e-1 * conv2d(dnet, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv2_2')

        net = conv2d(net, f_dim, name='conv3_0')
        dnet = conv2d(net, f_dim // 2, kernel_size=[3, 3], stride=[1, 1], name='conv3_1')
        net += 1e-1 * conv2d(dnet, f_dim, kernel_size=[3, 3], stride=[1, 1], name='conv3_2')

    net = tf.reshape(net, [-1, output_size // 16 * output_size // 16 * f_dim])
    logits = fully_connected(net, 1, activation_fn=None, is_spectral_norm=True, update_collection=None, name='fc4')
    logits = tf.squeeze(logits, -1)

    return logits
