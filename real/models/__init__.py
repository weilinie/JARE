import tensorflow as tf
from models import (
    conv4, conv4_nobn, dcgan4_nobn, resnet_v1, resnet_v2,
    # sn-gan
    conv4_sn, conv4_sn_nobn, dcgan4_sn_nobn, resnet_sn_v1, resnet_sn_v2
)


generator_dict = {
    'conv4': conv4.generator,
    'conv4_nobn': conv4_nobn.generator,
    'dcgan4_nobn': dcgan4_nobn.generator,
    'resnet_v1': resnet_v1.generator,
    'resnet_v2': resnet_v2.generator,

    # sn-gan
    'conv4_sn': conv4_sn.generator,
    'conv4_sn_nobn': conv4_sn_nobn.generator,
    'dcgan4_sn_nobn': dcgan4_sn_nobn.generator,
    'resnet_sn_v1': resnet_sn_v1.generator,
    'resnet_sn_v2': resnet_sn_v2.generator,
}

discriminator_dict = {
    'conv4': conv4.discriminator,
    'conv4_nobn': conv4_nobn.discriminator,
    'dcgan4_nobn': dcgan4_nobn.discriminator,
    'resnet_v1': resnet_v1.discriminator,
    'resnet_v2': resnet_v2.discriminator,

    # sn-gan
    'conv4_sn': conv4_sn.discriminator,
    'conv4_sn_nobn': conv4_sn_nobn.discriminator,
    'dcgan4_sn_nobn': dcgan4_sn_nobn.discriminator,
    'resnet_sn_v1': resnet_sn_v1.discriminator,
    'resnet_sn_v2': resnet_sn_v2.discriminator,
}


def get_generator(model_name, scope='generator', **kwargs):
    model_func = generator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)


def get_discriminator(model_name, scope='discriminator', **kwargs):
    model_func = discriminator_dict[model_name]
    return tf.make_template(scope, model_func, **kwargs)
