import torch
from ops import *
from generators import *
from discriminators import *
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Todo
def can_loss():

   #builds optimizers and losses
    style_loss = nn.MultiLabelSoftMarginLoss()
    real_vs_fake = nn.BCELoss()

    # model.G                  = model.generator(model, model.z)
    # model.D, model.D_logits, model.D_c, model.D_c_logits     = model.discriminator(model,
    #                                                           model.inputs, reuse=False)
    # if model.experience_flag:
    #   try:
    #     model.experience_selection = tf.convert_to_tensor(random.sample(model.experience_buffer, 16))
    #   except ValueError:
    #     model.experience_selection = tf.convert_to_tensor(model.experience_buffer)
    #   model.G = tf.concat([model.G, model.experience_selection], axis=0)

    # model.D_, model.D_logits_, model.D_c_, model.D_c_logits_ = model.discriminator(model,
    #                                                           model.G, reuse=True)
    # model.d_sum = histogram_summary("d", model.D)
    # model.d__sum = histogram_summary("d_", model.D_)
    # model.d_c_sum = histogram_summary("d_c", model.D_c)
    # model.d_c__sum = histogram_summary("d_c_", model.D_c_)
    # model.G_sum = image_summary("G", model.G)

    # correct_prediction = tf.equal(tf.argmax(model.y,1), tf.argmax(model.D_c,1))
    # model.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # true_label = tf.random_uniform(tf.shape(model.D),.8, 1.2)
    # false_label = tf.random_uniform(tf.shape(model.D_), 0.0, 0.3)

    # model.d_loss_real = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(model.D_logits, true_label * tf.ones_like(model.D)))

    # model.d_loss_fake = tf.reduce_mean(
    #   sigmoid_cross_entropy_with_logits(model.D_logits_, false_label * tf.ones_like(model.D_)))

    # model.d_loss_class_real = tf.reduce_mean(
    #   tf.nn.softmax_cross_entropy_with_logits(logits=model.D_c_logits, labels=model.smoothing * model.y))

    # # if classifier is set, then use the classifier, o/w use the clasification layers in the discriminator
    # if model.style_net_checkpoint is None:
    #   model.g_loss_class_fake = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits=model.D_c_logits_,
    #       labels=(1.0/model.y_dim)*tf.ones_like(model.D_c_)))
    # else:
    #   model.classifier = model.make_style_net(model.G)
    #   model.g_loss_class_fake = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits=model.classifier,
    #       labels=(1.0/model.y_dim)*tf.ones_like(model.D_c_)))

    # model.g_loss_fake = -tf.reduce_mean(tf.log(model.D_))

    # model.d_loss = model.d_loss_real + model.d_loss_class_real + model.d_loss_fake
    # model.g_loss = model.g_loss_fake + model.lamb * model.g_loss_class_fake

    # model.d_loss_real_sum       = scalar_summary("d_loss_real", model.d_loss_real)
    # model.d_loss_fake_sum       = scalar_summary("d_loss_fake", model.d_loss_fake)
    # model.d_loss_class_real_sum = scalar_summary("d_loss_class_real", model.d_loss_class_real)
    # model.g_loss_class_fake_sum = scalar_summary("g_loss_class_fake", model.g_loss_class_fake)
    # model.g_loss_sum = scalar_summary("g_loss", model.g_loss)
    # model.d_loss_sum = scalar_summary("d_loss", model.d_loss)
    # model.d_sum = merge_summary(
    #     [model.z_sum, model.d_sum, model.d_loss_real_sum, model.d_loss_sum,
    #     model.d_loss_class_real_sum, model.g_loss_class_fake_sum])
    # model.g_sum = merge_summary([model.z_sum, model.d__sum,
    #   model.G_sum, model.d_loss_fake_sum, model.g_loss_sum])

    # model.g_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    # model.d_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)

    # t_vars = tf.trainable_variables()
    # d_vars = [var for var in t_vars if 'd_' in var.name]
    # g_vars = [var for var in t_vars if 'g_' in var.name]

    # d_update = model.d_opt.minimize(model.d_loss, var_list=d_vars)
    # g_update = model.g_opt.minimize(model.g_loss, var_list=g_vars)

    # return d_update, g_update, [model.d_loss, model.g_loss], [model.d_sum, model.g_sum]

    return 0


def wgan_loss():
    return 0