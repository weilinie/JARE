import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os
import time

from utils import kde
from ops import *


tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parameters
learning_rate = 1e-3
reg_param = 10.
batch_size = 128
x_dim = 2
z_dim = 2
sigma = 0.7
mu = 2
method = 'advext'  # ['conopt', 'simgd', 'simregg', 'simregd', 'jare']
divergence = 'JS'  # ['standard', 'JS', 'indicator', 'wgan']
opt_type = 'sgd'  # ['sgd', 'rmsprop', 'adam']
outdir = os.path.join('kde_Isotrlin', time.strftime("%Y%m%d"), '{}'.format(divergence), 'std{}'.format(sigma),
                      '{}_{}_bs{}_std{}_reg{}_lr{}_{}_mu{}'.format(method, divergence, batch_size, sigma,
                                                                   reg_param, learning_rate, opt_type, mu))
sumdir = os.path.join('summary_Isotrlin', time.strftime("%Y%m%d"), '{}'.format(divergence), 'std{}'.format(sigma),
                      '{}_{}_bs{}_std{}_reg{}_lr{}_{}_mu{}'.format(method, divergence, batch_size, sigma,
                                                                   reg_param, learning_rate, opt_type, mu))
niter = 15000
n_save = 500
n_print = 100
bbox = [-2, 2, -2 + mu, 2 + mu]

# Target distribution
mus = np.vstack([0, mu] for _ in range(batch_size))
x_real = mus + sigma * tf.random_normal([batch_size, x_dim])

generator = tf.make_template('generator', generator4Gaussian_func1)
discriminator = tf.make_template('discriminator', discriminator4Gaussian_func1)

# g and d output
z = sigma * tf.random_normal([batch_size, z_dim])
x_fake = generator(z, x_dim)
d_out_real = discriminator(x_real)
d_out_fake = discriminator(x_fake)

d_loss, g_loss = compute_loss(d_out_real, d_out_fake, divergence)

# collect two sets of trainable variables
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

d_loss_tot, g_loss_tot, train_op, reg, d_grad_norm, g_grad_norm = \
    compute_gradients(d_loss, d_vars, g_loss, g_vars, opt_type, learning_rate, reg_param, method)

summary_op = tf.summary.merge([
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/g_loss", g_loss),
    tf.summary.scalar("loss/reg", reg),
    tf.summary.scalar("loss/d_loss_tot", d_loss_tot),
    tf.summary.scalar("loss/g_loss_tot", g_loss_tot),

    tf.summary.scalar("grad/d_grad_norm", d_grad_norm),
    tf.summary.scalar("grad/g_grad_norm", g_grad_norm),

])

print("Using the optimizer: {}".format(method))

# initialize and run
sess = tf.Session()
train_writer = tf.summary.FileWriter(sumdir, sess.graph)
sess.run(tf.global_variables_initializer())

if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Training: {}_{}_bs{}_mu{}_std{}_reg{}_lr{}'.format(
    method, divergence, batch_size, mu, sigma, reg_param, learning_rate))
ztest = [sigma * np.random.randn(batch_size, z_dim) for i in range(10)]

# generate real samples
x_real_out = np.concatenate([sess.run(x_real)])
init_g = sess.run(g_vars[0])
init_d = sess.run(d_vars[0])
print('initial theta: {}'.format(init_d))
print('initial phi: {}'.format(init_g))
kde(x_real_out[:, 0], x_real_out[:, 1], bbox=bbox, save_file=os.path.join(outdir, 'real.png'))
for i in range(niter):
    if i % n_print == 0:
        d_loss_out, g_loss_out, summary_str = sess.run([d_loss, g_loss, summary_op])
        train_writer.add_summary(summary_str, i)
        print('iters = %d, d_loss = %.4f, g_loss = %.4f' % (i, d_loss_out, g_loss_out))
    if i % n_save == 0:
        x_out = np.concatenate([sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
        kde(x_out[:, 0], x_out[:, 1], bbox=bbox, save_file=os.path.join(outdir, '%d.png' % i))
    sess.run(train_op)

sess.close()
