import tensorflow as tf

from libs.ops import train_opt


# Simultaneous gradient steps
class SimGDOptimizer(object):
    def __init__(self, opt_type, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self._sgd = train_opt(opt_type, learning_rate, beta1, beta2)
        self._eps = eps

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Gradient updates
        reg_grads = list(zip(grads, variables))

        train_op = self._sgd.apply_gradients(reg_grads)

        return [train_op]


# Alternating gradient steps
class AltGDOptimizer(object):
    def __init__(self, opt_type, learning_rate, beta1=0.9, beta2=0.999, d_steps=1, g_steps=1):
        self._d_sgd = train_opt(opt_type, learning_rate, beta1, beta2)
        self._g_sgd = train_opt(opt_type, learning_rate)
        self._d_steps = d_steps
        self._g_steps = g_steps

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        d_steps = self._d_steps
        g_steps = self._g_steps

        # Select train_op
        train_op_g = self._g_sgd.minimize(g_loss, var_list=g_vars)
        train_op_d = self._d_sgd.minimize(d_loss, var_list=d_vars)

        return [train_op_g] * g_steps + [train_op_d] * d_steps


class JAREOptimizer(object):
    def __init__(self, opt_type, learning_rate, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.optimizer = train_opt(opt_type, learning_rate, beta1, beta2)
        self._eps = eps
        self._alpha = alpha

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        d_grad_norm = tf.global_norm(d_grads)
        g_grad_norm = tf.global_norm(g_grads)
        reg_g = 0.5 * tf.square(d_grad_norm)
        reg_d = 0.5 * tf.square(g_grad_norm)

        # Jacobian times gradient
        Jgrads_g = tf.gradients(reg_g, g_vars)
        Jgrads_d = tf.gradients(reg_d, d_vars)

        # improved gradient regularization for g and d separately
        apply_g_vec = [
            (g + self._alpha * Jg, v)
            for (g, Jg, v) in zip(g_grads, Jgrads_g, g_vars) if Jg is not None
        ]
        apply_d_vec = [
            (g + self._alpha * Jg, v)
            for (g, Jg, v) in zip(d_grads, Jgrads_d, d_vars) if Jg is not None
        ]
        apply_vec = apply_d_vec + apply_g_vec
        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]


# Consensus optimization, method presented in the paper "The Numerics of GANs"
class ConsensusOptimizer(object):
    def __init__(self, opt_type, learning_rate, alpha=0.1, beta1=0.9, beta2=0.999,  eps=1e-8):
        self.optimizer = train_opt(opt_type, learning_rate, beta1, beta2)
        self._eps = eps
        self._alpha = alpha

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Reguliarizer
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads
        )
        # Jacobian times gradiant
        Jgrads = tf.gradients(reg, variables)

        # Gradient updates
        apply_vec = [
             (g + self._alpha * Jg, v)
             for (g, Jg, v) in zip(grads, Jgrads, variables) if Jg is not None
        ]

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]


# Try to stabilize training by gradient clipping (suggested by reviewer)
class ClipOptimizer(object):
    def __init__(self, opt_type, learning_rate, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.optimizer = train_opt(opt_type, learning_rate, beta1, beta2)
        self._eps = eps
        self._alpha = alpha

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Merge variable and gradient lists
        d_grads, _ = tf.clip_by_global_norm(d_grads, self._alpha)
        g_grads, _ = tf.clip_by_global_norm(g_grads, self._alpha)

        variables = d_vars + g_vars
        grads = d_grads + g_grads

        # Jacobian times gradiant
        # Gradient updates
        apply_vec = list(zip(grads, variables))

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]


# Like ConsensusOptimizer, but only take regularizer for discriminator (suggested by reviewer)
class SmoothingOptimizer(object):
    def __init__(self, opt_type, learning_rate, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.optimizer = train_opt(opt_type, learning_rate, beta1, beta2)
        self._eps = eps
        self._alpha = alpha

    def conciliate(self, d_loss, g_loss, d_vars, g_vars, global_step=None):
        # Compute gradients
        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)

        # Reguliarizer
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g))  for g in g_grads
        )

        # Jacobian times gradiant
        Jgrads = tf.gradients(reg, d_vars)
        # Gradient updates
        apply_vec = [
             (g + self._alpha * Jg, v)
             for (g, Jg, v) in zip(d_grads, Jgrads, d_vars) if Jg is not None
        ]
        apply_vec += list(zip(g_grads, g_vars))

        train_op = self.optimizer.apply_gradients(apply_vec)

        return [train_op]
