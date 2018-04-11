# model.py
# 'Driver' of the GAN
# Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py 
# and  https://github.com/mlberkeley/Creative-Adversarial-Networks/blob/master/model.py
# 9/4/18

import argparse
import os
from generators import *
from utils import *
from ops import *
from discriminators import *
import random
import torch
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable



# GAN
# can be DCGAN, CAN, WGAN etc
class GAN:

    def __init__(self, options):
        self.dataset = options.dataset
        self.dataroot = options.dataroot
        self.workers = options.workers 
        self.type = options.gan_type
        self.batch_size = options.batch_size
        self.pow = options.pow
        self.image_size = options.image_size
        self.lr = options.lr
        self.z_noise = options.z_noise
        self.y_dim = options.y_dim
        self.beta1 = options.beta1
        self.channels = options.channels
        self.num_gen_filters = options.num_gen_filters
        self.num_disc_filters = options.num_disc_filters
        self.num_epochs = options.num_epochs
        self.cuda = options.cuda
        self.num_gpu = options.num_gpu
        self.gen_path = options.gen_path
        self.disc_path = options.disc_path
        self.out_folder = options.out_folder
        self.manual_seed = options.manual_seed

 


    # custom weights initialization called on self.generator and self.discriminator
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self):
        data = get_dataset(self.dataroot)
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                         shuffle=True, num_workers=int(self.workers))

        # Set the type of GAN
        if self.type == "dcgan": #isize, nz, nc, ndf, ngpu, n_extra_layers=0)
            self.generator = DcganGenerator(self.z_noise, self.image_size, self.channels, self.num_gen_filters, self.pow)
            self.discriminator = DcganDiscriminator(self.image_size,self.channels, self.num_disc_filters, self.pow)
            criterion = nn.BCELoss()

        elif self.type == "can":
            self.generator = CanGenerator
            self.discriminator = CanDiscriminator
            criterion = can_loss()
        
        elif self.type == "wgan":
            self.generator = CanGenerator
            self.discriminator = CanDiscriminator
            criterion = wgan_loss()

        self.discriminator.apply(self.weights_init)
        if self.disc_path != '':
            self.discriminator.load_state_dict(torch.load(self.disc_path))
        print(self.discriminator)
        print(self.generator)
        criterion = nn.BCELoss()

        inp = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.z_noise, 1, 1)
        fixed_noise = torch.FloatTensor(self.batch_size, self.z_noise, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        if self.cuda:
            self.discriminator.cuda()
            self.generator.cuda()
            criterion.cuda()
            inp, label = inp.cuda(), label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        fixed_noise = Variable(fixed_noise)

        # setup optimizer
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        for epoch in range(self.num_epochs):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.discriminator.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                if self.cuda:
                    real_cpu = real_cpu.cuda()
                inp.resize_as_(real_cpu).copy_(real_cpu)
                label.resize_(batch_size).fill_(real_label)
                inputv = Variable(inp)
                labelv = Variable(label)
               
                output = self.discriminator(inputv)
               
                errD_real = criterion(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                noise.resize_(batch_size, self.z_noise, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = self.generator(noisev)
                labelv = Variable(label.fill_(fake_label))
                output = self.discriminator(fake.detach())
                errD_fake = criterion(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                output = self.discriminator(fake)
                errG = criterion(output, labelv)
                errG.backward()
                D_G_z2 = output.data.mean()
                optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, self.num_epochs, i, len(dataloader),
                        errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples_%03d.png' % (self.out_folder,i),
                            normalize=True)
                    fake = self.generator(fixed_noise)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_epoch_%03d.png' % (self.out_folder, epoch),
                            normalize=True)

            # do checkpointing
            torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (self.out_folder, epoch))
            torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (self.out_folder, epoch))