import time
import os
import tensorflow as tf
from tqdm import tqdm

from metrics.classifier_metrics_impl import get_graph_def_from_disk
from metrics.fid_score import get_fid_function
from metrics import fid
from metrics.inception_score import InceptionScore
from optimizer import (
    JAREOptimizer, ConsensusOptimizer, SimGDOptimizer, AltGDOptimizer, ClipOptimizer, SmoothingOptimizer
)
from utils import *

INCEPTION_FROZEN_GRAPH = 'inception_frozen/inceptionv1_for_inception_score.pb'


def train(generator, discriminator, x_real, config):
    batch_size = config['batch_size']
    output_size = config['output_size']
    c_dim = config['c_dim']
    z_dim = config['z_dim']

    # TODO: fix that this has to be run before all other graph building ops
    if config['is_inception_scores']:
        print("Inception score is calculated...")
        inception_scorer = InceptionScore(config['inception_dir'])

    x_real = 2. * x_real - 1.
    z = tf.random_normal([batch_size, z_dim])
    x_fake = generator(z)
    x_fake_test = generator(z, is_training=False)
    d_out_real = discriminator(x_real)
    d_out_fake = discriminator(x_fake)

    # import inception graph into current graph
    if config['fid_type'] == 1:
        print("FID is calculated by using the same number of real and generated images...")
        x_real_scale = tf.cast(tf.clip_by_value((x_real + 1.) * 127.5, 0.0, 255.0), tf.float32)
        x_fake_test_scale = tf.cast(tf.clip_by_value((x_fake_test + 1.) * 127.5, 0.0, 255.0), tf.float32)
        inception_graph_name = os.path.join(config['inception_dir'], INCEPTION_FROZEN_GRAPH)
        inception_graph = get_graph_def_from_disk(inception_graph_name)
        fid_eval = get_fid_function(
            x_real_scale, x_fake_test_scale, num_eval_images=64 * 200,
            image_range="0_255", inception_graph=inception_graph)

    # using pretrained real image statistics to get fid
    elif config['fid_type'] == 2:
        print("FID is calculated using precalculated real image statistics...")
        inception_file = fid.check_or_download_inception(config['inception_dir'])
        fid.create_inception_graph(inception_file)

    else:
        print("FID is not calculated!")
        pass

    # GAN / Divergence type
    g_loss, d_loss = get_losses(d_out_real, d_out_fake, x_real, x_fake, discriminator, config)

    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    # Optimizer
    optimizer = get_optimizer(config, global_step)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    train_ops = optimizer.conciliate(d_loss, g_loss, d_vars, g_vars, global_step=global_step)

    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)

    # Summaries
    summaries = [
        tf.summary.scalar('loss/discriminator', d_loss),
        tf.summary.scalar('loss/generator', g_loss),
        tf.summary.scalar('loss/Wall_clock_time', Wall_clock_time),
    ]
    summary_op = tf.summary.merge(summaries)

    # Inception scores
    inception_scores = tf.placeholder(tf.float32)
    inception_mean, inception_var = tf.nn.moments(inception_scores, [0])
    inception_summary_op = tf.summary.merge([
        tf.summary.scalar('inception_score/mean', inception_mean),
        tf.summary.scalar('inception_score/std', tf.sqrt(inception_var)),
        tf.summary.scalar('inception_score/Wall_clock_time', Wall_clock_time),
        tf.summary.histogram('inception_score/histogram', inception_scores)
    ])
    # fid score 2
    fid_score = tf.placeholder(tf.float32)
    fid_sum_op = tf.summary.scalar("last_computed_FID2", fid_score)

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=config['log_dir'], global_step=global_step,
        summary_op=summary_op, save_model_secs=3600, save_summaries_secs=300
    )

    z_test_np = np.random.randn(batch_size, z_dim)

    print('training...')

    with sv.managed_session() as sess:
        # Show real data
        samples = sess.run(x_real)
        samples = (samples + 1.) * (255. / 2)
        save_images(samples[:64], config['sample_dir'], 'real.png')

        progress = tqdm(range(config['nsteps']))

        for batch_idx in progress:
            if sv.should_stop():
                break

            niter = sess.run(global_step)

            t0 = time.time()
            # Train
            for train_op in train_ops:
                sess.run(train_op)
            t1 = time.time()

            sess.run(update_Wall_op, feed_dict={time_diff: t1 - t0})

            d_loss_out, g_loss_out = sess.run([d_loss, g_loss])

            progress.set_description('Loss_g: %4.4f, Loss_d: %4.4f' % (g_loss_out, d_loss_out))
            sess.run(global_step_op)

            if np.mod(niter, config['ntest']) == 0:
                # Test
                samples = sess.run(x_fake_test, feed_dict={z: z_test_np})
                samples = np.clip((samples + 1.) * 127.5, 0.0, 255.0).astype(np.uint8)

                save_images(samples[:64], os.path.join(config['sample_dir'], 'samples'),
                            'train_{:06d}.png'.format(niter))

                # FID without precalculated stats
                if config['fid_type'] == 1:
                    fid_score_np, fid_sum_out = fid_eval(sess)
                    sv.summary_computed(sess, fid_sum_out)

                # FID with precalculated stats
                elif config['fid_type'] == 2:
                    fid_path = os.path.join(config['data_dir'], 'fid_stats', 'fid_stats_cifar10_train.npz')
                    fid_score_np = get_fid_score(sess, fid_path, x_fake_test, batch_size)
                    fid_sum_op_out = sess.run(
                        fid_sum_op, feed_dict={fid_score: fid_score_np})
                    sv.summary_computed(sess, fid_sum_op_out)

                else:
                    pass

                # Inception scores
                if config['is_inception_scores']:
                    inception_scores_np = get_inception_score(sess, inception_scorer, x_fake_test, batch_size)
                    inception_score_mean, inception_score_std = np.mean(inception_scores_np), np.std(
                        inception_scores_np)
                    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))
                    inception_scores_sum_out = sess.run(
                        inception_summary_op, feed_dict={inception_scores: inception_scores_np})
                    sv.summary_computed(sess, inception_scores_sum_out)


def get_losses(d_out_real, d_out_fake, x_real, x_fake, discriminator, config):
    batch_size = config['batch_size']
    gan_type = config['gan_type']

    if gan_type == 'standard':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.ones_like(d_out_fake)
        ))
    elif gan_type == 'JS':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake
    elif gan_type == 'KL':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.reduce_mean(-d_out_fake)
    elif gan_type == 'tv':
        d_loss = tf.reduce_mean(tf.tanh(d_out_fake) - tf.tanh(d_out_real))
        g_loss = tf.reduce_mean(-tf.tanh(d_out_fake))
    elif gan_type == 'indicator':
        d_loss = tf.reduce_mean(d_out_fake - d_out_real)
        g_loss = tf.reduce_mean(-d_out_fake)
    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % gan_type)

    return g_loss, d_loss


def get_optimizer(config, global_step):
    optimizer_name = config['optimizer']

    reg_param = config['reg_param']
    learning_rate = config['learning_rate']
    opt_type = config['opt_type']
    nsteps = config['nsteps']
    beta1 = config['beta1']
    beta2 = config['beta2']

    learning_rate_decayed = learning_rate  # tf.train.exponential_decay(learning_rate, global_step, nsteps, 0.01)

    if optimizer_name == 'simgd':
        optimizer = SimGDOptimizer(opt_type, learning_rate_decayed, beta1=beta1, beta2=beta2)
    elif optimizer_name == 'altgd':
        optimizer = AltGDOptimizer(opt_type, learning_rate_decayed, beta1=beta1, beta2=beta2,
                                   g_steps=config['altgd_gsteps'], d_steps=config['altgd_dsteps'])
    elif optimizer_name == 'jare':
        optimizer = JAREOptimizer(opt_type, learning_rate_decayed, alpha=reg_param, beta1=beta1, beta2=beta2)
    elif optimizer_name == 'conopt':
        optimizer = ConsensusOptimizer(opt_type, learning_rate_decayed, alpha=reg_param, beta1=beta1, beta2=beta2)
    elif optimizer_name == 'clip':
        optimizer = ClipOptimizer(opt_type, learning_rate_decayed, alpha=reg_param, beta1=beta1, beta2=beta2)
    elif optimizer_name == 'smooth':
        optimizer = SmoothingOptimizer(opt_type, learning_rate_decayed, alpha=reg_param, beta1=beta1, beta2=beta2)
    else:
        raise Exception("Unknown optimizer name")

    return optimizer


def get_inception_score(sess, inception_scorer, x_fake_test, batch_size):
    all_samples = []
    num_images_to_eval = 50000
    num_batches = num_images_to_eval // batch_size + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    for _ in range(num_batches):
        all_samples.append(sess.run(x_fake_test))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:num_images_to_eval]
    all_samples = np.clip((all_samples + 1.) * 127.5, 0.0, 255.0).astype(np.uint8)
    return inception_scorer.get_inception_score(sess, list(all_samples))


def get_fid_score(sess, fid_file, x_fake, batch_size):
    all_samples = []
    num_images_to_eval = 10000
    num_batches = num_images_to_eval // batch_size + 1
    print("Calculating FID. Sampling {} images...".format(num_images_to_eval))
    for _ in range(num_batches):
        all_samples.append(sess.run(x_fake))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:num_images_to_eval]
    all_samples = np.clip((all_samples + 1.) * 127.5, 0.0, 255.0).astype(np.uint8)
    mu_gen, sigma_gen = fid.calculate_activation_statistics(all_samples, sess, batch_size=100)  # 50

    f = np.load(fid_file)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]

    return fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
