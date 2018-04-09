# model.py
# 'Driver' of the GAN
# Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py 
# and  https://github.com/mlberkeley/Creative-Adversarial-Networks/blob/master/model.py
# 9/4/18

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


# parser.add_argument('--z_noise', type=int, default=100, help='size of the latent z vector')
# #check what these are
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)

# parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--gen', default='', help="path to netG (to continue training)")
# parser.add_argument('--dis', default='', help="path to netD (to continue training)")
# parser.add_argument('--out_folder', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manual_seed', type=int, help='manual seed')


# GAN
# can be DCGAN, CAN, WGAN etc
class GAN:

    def __init__(self, options):
        self.dataset = options.dataset
        self.dataroot = options.dataroot
        self.workers = options.workers 
        self.type = options.gan_type
        self.batch_size = options.batch_size
        self.image_size = options.image_size
        self.z_noise = options.z_noise
        self.num_gen_filters = options.num_gen_filters
        self.num_disc_filters = options.num_disc_filters
        self.num_epochs = options.num_epochs
        self.cuda = options.cuda
        self.num_gpu = options.num_gpu
        self.gen_path = options.gen_path
        self.disc_path = options.disc_path
        self.out_folder = options.out_folder
        self.manual_seed = options.manual_seed
