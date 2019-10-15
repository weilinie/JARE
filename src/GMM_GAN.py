import tensorflow as tf

import numpy as np
import os
import time
import argparse

from src.utils import kde
from src.ops import *

tf.reset_default_graph()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--reg', type=float, default=1000, help='reg_param')
parser.add_argument('--mu', type=float, default=4, help='mu')
parser.add_argument('--sigma', type=float, default=0.06, help='sigma')
parser.add_argument('--method', type=str, default='conopt', help='reg_method')
config, unparsed = parser.parse_known_args()

# Parameters
learning_rate = 1e-4
reg_param = config.reg
batch_size = 128  # [512, 128]
z_dim = 64  # [256, 64]
sigma = config.sigma  # [0.02, 0.1]
mu = config.mu
sigma_noise = np.sqrt(mu)
method = config.method  # ['conopt', 'simgd', 'simregg', 'simregd', 'jare']
divergence = 'JS'  # ['standard', 'JS', 'indicator', 'wgan']
opt_type = 'rmsprop'  # ['sgd', 'rmsprop', 'adam']
model = 'model1'  # ['model1', 'model2']
outdir = os.path.join('gifs', time.strftime("%Y%m%d"), '{}'.format(divergence), 'mu{}'.format(mu),
                      '{}_{}_bs{}_zdim{}_reg{}_lr{}_{}_{}_mu{}_std{}'.format(
                          method, divergence, batch_size, z_dim, reg_param, learning_rate,
                          opt_type, model, mu, sigma))
sumdir = os.path.join('summaries', time.strftime("%Y%m%d"), '{}'.format(divergence), 'mu{}'.format(mu),
                      '{}_{}_bs{}_zdim{}_reg{}_lr{}_{}_{}_mu{}_std{}'.format(
                          method, divergence, batch_size, z_dim, reg_param, learning_rate,
                          opt_type, model, mu, sigma))
niter = 20000  # 6000 dimensionality
n_save = 500
n_print = 100
bbox = [-mu, mu, -mu, mu]

# Target distribution
mus = np.vstack([mu/2.*np.cos(2*np.pi*k/8), mu/2.*np.sin(2*np.pi*k/8)] for k in range(batch_size))
x_real = mus + sigma*tf.random_normal([batch_size, 2])

if model == 'model1':
    generator = tf.make_template('generator', generator_func1)
    discriminator = tf.make_template('discriminator', discriminator_func1)
elif model == 'model2':
    generator = tf.make_template('generator', generator_func2)
    discriminator = tf.make_template('discriminator', discriminator_func2)
else:
    raise NotImplementedError


# g and d output
z = tf.random_normal([batch_size, z_dim], stddev=sigma_noise)
x_fake = generator(z)
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

    tf.summary.scalar("grad/d_gnrad_norm", d_grad_norm),
    tf.summary.scalar("grad/g_grad_norm", g_grad_norm)
])

# initialize and run
sess = tf.Session()
train_writer = tf.summary.FileWriter(sumdir, sess.graph)
sess.run(tf.global_variables_initializer())

if not os.path.exists(outdir):
    os.makedirs(outdir)

print('Training: {}_{}_bs{}_zdim{}_reg{}_mu{}_std{}'.format(
    method, divergence, batch_size, z_dim, reg_param, mu, sigma))
ztest = [sigma_noise*np.random.randn(batch_size, z_dim) for i in range(10)]

for i in range(niter):
    if i % n_print == 0:
        d_loss_out, g_loss_out, summary_str = sess.run([d_loss, g_loss, summary_op])
        train_writer.add_summary(summary_str, i)
        print('iters = %d, d_loss = %.4f, g_loss = %.4f' % (i, d_loss_out, g_loss_out))
    if i % n_save == 0:
        x_out = np.concatenate([sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
        kde(x_out[:, 0], x_out[:, 1], bbox=bbox, save_file=os.path.join(outdir,'%d.png' % i))
    sess.run(train_op)

sess.close()
