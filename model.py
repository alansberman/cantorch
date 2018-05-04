# model.py
# 'Driver' of the GAN
# Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py 
# and  https://github.com/mlberkeley/Creative-Adversarial-Networks/blob/master/model.py
# and https://github.com/martinarjovsky/WassersteinGAN etc
# 9/4/18

import argparse
import os
from generators import *
from discriminators import *
from ops import *
from utils import *
import random
import torch
import numpy as np
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
        self.power = options.power
        self.image_size = options.image_size
        self.lr = options.lr
        self.gan_type = options.gan_type
        self.z_noise = options.z_noise
        self.y_dim = options.y_dim
        self.beta1 = options.beta1
        self.channels = options.channels
        self.num_gen_filters = options.num_gen_filters
        self.num_disc_filters = options.num_disc_filters
        self.num_epochs = options.num_epochs
        self.cuda = options.cuda
        self.disc_iterations = options.disc_iterations
        self.num_gpu = options.num_gpu
        self.lower_clamp = options.lower_clamp
        self.upper_clamp = options.upper_clamp
        self.gen_path = options.gen_path
        self.disc_path = options.disc_path
        self.out_folder = options.out_folder
        self.manual_seed = options.manual_seed

        self.train_history =  {}
        self.train_history['disc_loss'] = []
        self.train_history['gen_loss'] = []
        

    # custom weights initialization called on self.generator and self.discriminator
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train_discriminator(self,data_iterator,criterion,inp,label,noise,style_criterion=None):
        '''
        Trains the GAN discriminator
        includes style classification loss if creating a CAN

        '''

        # Image style/class labels
        style_labels = torch.LongTensor(self.batch_size)
        if self.cuda:
            style_labels = style_labels.cuda()

        # Clamp as per WGAN (better training)
        for param in self.discriminator.parameters():
            param.requires_grad = True
            if self.type == 'wgan':
                param.data.clamp_(self.lower_clamp,self.upper_clamp)
            
        data = data_iterator.next()

        # train with real
        self.discriminator.zero_grad()

        real_image, batch_styles = data

        # Get the style/class labels for the batch of (real) images
        style_labels = style_labels.copy_(batch_styles)
        style_labels = Variable(style_labels.copy_(batch_styles))
        batch_size = real_image.size(0)
        if self.cuda:
            real_image = real_image.cuda()
        inp.resize_as_(real_image).copy_(real_image)
        # Fill with real_value (1)
        label.resize_(batch_size).fill_(1)
        input_var = Variable(inp)
        label_var = Variable(label)

        if self.type == 'can':
            output, output_styles = self.discriminator(input_var)   
        else:
            output = self.discriminator(input_var)
        
        if self.type == "can":
            #Train D with real images and get the style class loss
            err_disc_style = style_criterion(output_styles, style_labels)
            err_disc_style.backward(retain_graph=True)

        err_disc_real = criterion(output, label_var)
        err_disc_real.backward(retain_graph=True)
        disc_x = output.data.mean()

        # train with fake
        noise.resize_(self.batch_size, self.z_noise, 1, 1).normal_(0, 1)
        noise_var = Variable(noise) # Tried to freeze gen with volatile=True but that failed
        fake = self.generator(noise_var)
        label_var = Variable(label.fill_(0))
        if self.type == 'can':
            output, batch_styles = self.discriminator(fake)
        else:
            output = self.discriminator(fake)
        err_disc_fake = criterion(output, label_var)
        err_disc_fake.backward(retain_graph=True)
        disc_gen_z_1 = output.data.mean()
        
        disc_loss = err_disc_real + err_disc_fake 
        # Add style loss to overall loss if a CAN
        if self.type == 'can':
            disc_loss += err_disc_style
        self.disc_optimizer.step()
        return disc_loss,fake,disc_x,disc_gen_z_1,real_image

    def train_generator(self,noise,label,fake,criterion,gen_style_labels=None,style_criterion=None):
        '''
        Trains the GAN generator
        includes style classification loss if creating a CAN

        '''
  
        self.generator.zero_grad()     


        label_var = Variable(label.fill_(1))  # fake labels are real for generator cost
        if self.type == 'can':
            output, batch_labels = self.discriminator(fake) #can't say detach() for some reason
        else:
            output = self.discriminator(fake)

        # Normal GAN loss
        err_gen = criterion(output, label_var)
        err_gen.backward(retain_graph=True)
        gen_loss = err_gen
        
        # Add style classification loss
        # but use dummy labels
        if self.type == 'can':
            err_gen_style = style_criterion(batch_labels,gen_style_labels)
            err_gen_style.backward()
            gen_loss +=  err_gen_style
        
        disc_gen_z_2 = output.data.mean()
        self.gen_optimizer.step()
        return gen_loss, disc_gen_z_2

    def train(self):
        print(torch.cuda.get_device_name(0),"is current GPU")

        # Get dataset
        data = get_dataset(self.dataroot)
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                         shuffle=True, num_workers=int(self.workers),drop_last=True)

        # Set the type of GAN
        if self.type == "dcgan": 
            self.generator = DcganGenerator(self.z_noise, self.image_size, self.channels, self.num_gen_filters, self.power)
            self.discriminator = DcganDiscriminator(self.image_size,self.channels, self.num_disc_filters, self.power)
            criterion = nn.BCELoss()

        elif self.type == "can":
            if self.image_size == 64:
                self.generator = Can64Generator(self.z_noise, self.image_size, self.channels, self.num_gen_filters, self.power)
                self.discriminator = Can64Discriminator(self.image_size,self.channels,self.y_dim, self.num_disc_filters, self.power)
            else:
                self.generator = CanGenerator(self.z_noise, self.image_size, self.channels, self.num_gen_filters, self.power)
                self.discriminator = CanDiscriminator(self.image_size,self.channels,self.y_dim, self.num_disc_filters, self.power)
            # 2 losses
            criterion = nn.BCELoss()
            style_criterion = nn.CrossEntropyLoss()
            if self.cuda:
                style_criterion.cuda()

        
        elif self.type == "wgan": #todo
            self.generator = CanGenerator
            self.discriminator = CanDiscriminator
            criterion = wgan_loss()

        self.discriminator.apply(self.weights_init)
        self.generator.apply(self.weights_init) 

        if self.disc_path != '':
            self.discriminator.load_state_dict(torch.load(self.disc_path))
            
        print(self.discriminator)
        print(self.generator)

        # Placeholders
        inp = torch.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        noise = torch.FloatTensor(self.batch_size, self.z_noise, 1, 1)

        # Normalized noise
        fixed_noise = torch.FloatTensor(self.batch_size, self.z_noise, 1, 1).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)

        # Generator class/style labels
        gen_style_labels = torch.LongTensor(self.batch_size)

        # Dummy label as generator not trained on class/style labels
        gen_style_labels = gen_style_labels.fill_(1)

        # Cuda vars
        if self.cuda:
            self.discriminator.cuda()
            self.generator.cuda()
            criterion.cuda()
            inp, label = inp.cuda(), label.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            gen_style_labels = gen_style_labels.cuda()
          

        fixed_noise = Variable(fixed_noise)                   
        gen_style_labels = Variable(gen_style_labels)

     
        # setup optimizers
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # Actual training!
        for epoch in range(self.num_epochs):
            data_iterator = iter(dataloader)
            i = 0
            
            disc_loss_epoch =[]
            gen_loss_epoch = []
            
            # While more batches
            # Heavily inspired by https://github.com/martinarjovsky/WassersteinGAN
            while i < len(dataloader):  

                
                # Will set false in G
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                # # Freeze generator
                # for p in self.generator.parameters():
                #     p.requires_grad = False

                j=0
                # Train the discriminator disc_iterations times for every 1 generator training
                while j < self.disc_iterations: #and i < len(dataloader)
                    j += 1
                    if self.type=='can':
                        disc_loss,fake,disc_x,disc_gen_z_1, real_image = self.train_discriminator(data_iterator,criterion,inp,label,noise,style_criterion)
                    else:
                        disc_loss,fake,disc_x,disc_gen_z_1, real_image = self.train_discriminator(data_iterator,criterion,inp,label,noise)
                   

                # Train generator
                # Freeze disc
                for par in self.discriminator.parameters():
                    par.requires_grad = False

                # # Unfreeze generator
                # for item in self.generator.parameters():
                #     item.requires_grad = True

      
                if self.type=='can':
                    gen_loss, disc_gen_z_2 = self.train_generator(noise,label,fake,criterion,gen_style_labels,style_criterion)
                else:
                    gen_loss, disc_gen_z_2 = self.train_generator(noise,label,fake,criterion)
               
                # Update the train history
                disc_loss_epoch.append(disc_loss.data[0])
                gen_loss_epoch.append(gen_loss.data[0])

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, self.num_epochs, i, len(dataloader),
                        disc_loss.data[0], gen_loss.data[0], disc_x, disc_gen_z_1, disc_gen_z_2))
                if (i % 500 == 0) or i == (len(dataloader) -1):
                    vutils.save_image(real_image,
                            '%s/real_samples_epoch_%03d_%04d.png' % (self.out_folder,epoch,i),
                            normalize=True)
                    fake = self.generator(fixed_noise)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_epoch_%03d_%04d.png' % (self.out_folder, epoch,i),
                            normalize=True)
                i+=1

            # Metrics for Floydhub
            print('{{"metric": "disc_loss", "value": {:.4f}}}'.format(np.mean(disc_loss_epoch)))
            print('{{"metric": "gen_loss", "value": {:.4f}}}'.format(np.mean(gen_loss_epoch)))

            # Get the mean of the losses over the epoch for the loss graphs
            disc_loss_epoch = np.asarray(disc_loss_epoch)
            gen_loss_epoch = np.asarray(gen_loss_epoch)

            self.train_history['disc_loss'].append(np.mean(disc_loss_epoch))
            self.train_history['gen_loss'].append(np.mean(gen_loss_epoch))
            
            # do checkpointing
            torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (self.out_folder, epoch))
            torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (self.out_folder, epoch))
        
        get_loss_graphs(self.train_history,self.out_folder,self.gan_type)


