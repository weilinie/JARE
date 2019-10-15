import tensorflow as tf
from tensorflow.contrib import slim


##################################################################################
#  generator and discriminator for GMM
##################################################################################
# Model 1
def generator4Gaussian_func1(z, x_dim):
    phi = tf.get_variable(name='G_param', shape=[1, x_dim], initializer=tf.constant_initializer([0.05, mu + 0.05]))
    x = z + phi
    return x


def discriminator4Gaussian_func1(x):
    x_dim = x.shape[1].value
    theta = tf.get_variable(name='D_param', shape=[x_dim, 1], initializer=tf.constant_initializer([[0.05], [0.05]]))
    out = tf.matmul(x, theta)
    return out


##################################################################################
#  generator and discriminator for GMM
##################################################################################
# Model 1
def generator4gmm_func1(z):
    net = slim.fully_connected(z, 256)
    net = slim.fully_connected(net, 256)
    net = slim.fully_connected(net, 256)
    net = slim.fully_connected(net, 256)
    x = slim.fully_connected(net, 2, activation_fn=None)
    return x


def discriminator4gmm_func1(x):
    # Network
    net = slim.fully_connected(x, 256)
    net = slim.fully_connected(net, 256)
    net = slim.fully_connected(net, 256)
    net = slim.fully_connected(net, 256)
    logits = slim.fully_connected(net, 1, activation_fn=None)
    out = tf.squeeze(logits, -1)
    return out


# Model 2
def generator4gmm_func2(z, output_dim=2, n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x


def discriminator4gmm_func2(x, n_hidden=128, n_layer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(tf.divide(x, 4.0), slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d


def compute_loss(d_out_real, d_out_fake, divergence):
    # Loss
    if divergence == 'standard':
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
    elif divergence == 'JS':
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.ones_like(d_out_real)
        ))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
        ))
        d_loss = d_loss_real + d_loss_fake
        g_loss = -d_loss
    elif divergence == 'indicator':
        d_loss = tf.reduce_mean(d_out_real - d_out_fake)
        g_loss = -d_loss
    elif divergence == 'wgan':
        d_loss = tf.reduce_mean(d_out_fake - d_out_real)
        g_loss = tf.reduce_mean(-d_out_fake)
    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % divergence)

    return d_loss, g_loss


def compute_gradients(d_loss, d_vars, g_loss, g_vars, opt_type, learning_rate, reg_param, method):
    # optimization type
    if opt_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_type == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif opt_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    else:
        raise NotImplementedError

    # Compute gradients as list
    d_grads = tf.gradients(d_loss, d_vars)
    g_grads = tf.gradients(g_loss, g_vars)
    d_grad_norm = tf.global_norm(d_grads)
    g_grad_norm = tf.global_norm(g_grads)

    # Merge variable and gradient lists
    variables = d_vars + g_vars
    grads = d_grads + g_grads

    # apply different regularization methods
    if method == 'simgd':
        reg = 0
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        apply_vec = list(zip(grads, variables))
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'conopt':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, variables)
        # Gradient updates
        apply_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(grads, Jgrads, variables) if Jg is not None
        ]
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'simregg1':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, g_vars)
        # gradient-based regularization for g only
        apply_d_vec = list(zip(d_grads, d_vars))
        apply_g_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(g_grads, Jgrads, g_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'simregg2':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in d_grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, g_vars)
        # gradient-based regularization for g only
        apply_d_vec = list(zip(d_grads, d_vars))
        apply_g_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(g_grads, Jgrads, g_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'simregd1':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, d_vars)
        # gradient-based regularization for g only
        apply_g_vec = list(zip(g_grads, g_vars))
        apply_d_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(d_grads, Jgrads, d_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'simregd2':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in g_grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss + reg_param * reg
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, d_vars)
        # gradient-based regularization for g only
        apply_g_vec = list(zip(g_grads, g_vars))
        apply_d_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(d_grads, Jgrads, d_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'jare':
        reg_g = 0.5 * tf.square(d_grad_norm)
        reg_d = 0.5 * tf.square(g_grad_norm)
        # for back compatibility
        reg = reg_d + reg_g
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg_g
        d_loss_tot = d_loss + reg_param * reg_d
        # Jacobian times gradient
        Jgrads_g = tf.gradients(reg_g, g_vars)
        Jgrads_d = tf.gradients(reg_d, d_vars)
        # gradient-based regularization for g and d separately
        apply_g_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(g_grads, Jgrads_g, g_vars) if Jg is not None
        ]
        apply_d_vec = [
            (g + reg_param * Jg, v)
            for (g, Jg, v) in zip(d_grads, Jgrads_d, d_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    elif method == 'altregg':
        # Regularizer (asymptotically unbiased)
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in d_grads
        )
        # calculate the regularized d_loss and g_loss
        g_loss_tot = g_loss + reg_param * reg
        d_loss_tot = d_loss
        train_op = [optimizer.minimize(g_loss_tot, var_list=g_vars), optimizer.minimize(d_loss_tot, var_list=d_vars)]

    else:
        raise NotImplementedError

    return d_loss_tot, g_loss_tot, train_op, reg, d_grad_norm, g_grad_norm

