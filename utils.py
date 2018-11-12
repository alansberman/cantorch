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

    # Pytorch
    # To maximise, make negative
    # discriminator
 
    # model.g_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    # model.d_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)

    # t_vars = tf.trainable_variables()
    # d_vars = [var for var in t_vars if 'd_' in var.name]
    # g_vars = [var for var in t_vars if 'g_' in var.name]

    # d_update = model.d_opt.minimize(model.d_loss, var_list=d_vars)
    # g_update = model.g_opt.minimize(model.g_loss, var_list=g_vars)

    # return d_update, g_update, [model.d_loss, model.g_loss], [model.d_sum, model.g_sum]


def wgan_loss():
    return 0