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
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import torch.nn.init as init


import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import time


# GAN
# can be DCGAN, CAN, WGAN etc
class WGAN:

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
        self.gradient_penalty = options.gradient_penalty
       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_history =  {}
        self.train_history['disc_loss'] = []
        self.train_history['gen_loss'] = []
        self.train_history['disc_class_loss'] = []
        self.train_history['gen_class_loss'] = []
        

    # custom weights initialization called on self.generator and self.discriminator
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
    def weight_init(self,m):
        if isinstance(m, MyConvo2d): 
            if m.conv.weight is not None:
                if m.he_init:
                    init.kaiming_uniform_(m.conv.weight)
                else:
                    init.xavier_uniform_(m.conv.weight)
            if m.conv.bias is not None:
                init.constant_(m.conv.bias, 0.0)
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)



    def calc_gradient_penalty(self,netD, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(self.batch_size, int(real_data.nelement()/self.batch_size)).contiguous()
        alpha = alpha.view(self.batch_size, 3, 64, 64)
        alpha = alpha.to(self.device)
        
        fake_data = fake_data.view(self.batch_size, 3, 64, 64)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10 #10 = lambda
        return gradient_penalty
    
    def train(self):
        print("GPU/CPU:",torch.cuda.get_device_name(0))
        # Start timer
        model_file = open(self.out_folder+'/model_notes.txt',"w")
        start_time = time.time()
        training_notes_file = open(self.out_folder+'/training_notes.txt',"w")
        losses_file = open(self.out_folder+'/losses_notes.txt',"w")
        if self.dataset == 'cifar10':
            data = dset.CIFAR10(root= self.dataroot, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
            )
        
        if self.dataset == 'imagenet':
            data = dset.ImageFolder(root= self.dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
            )

        else:
            # Get dataset
            data = get_dataset(self.dataroot)
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                         shuffle=True, num_workers=int(self.workers),drop_last=True,pin_memory=True)

        # # Set the type of GAN
        if self.type == "dcgan": 
            if self.gradient_penalty:
                self.generator = GoodGenerator(64,64*64*3).to(self.device)
                self.discriminator = GoodDiscriminator(64).to(self.device)
                criterion = nn.BCELoss()
            else:
                self.generator = DcganGenerator(self.z_noise, self.channels, self.num_gen_filters).to(self.device)
                self.discriminator = WganDiscriminator(self.channels, self.num_disc_filters).to(self.device)
                criterion = nn.BCELoss()

        if self.type == "can":
            if self.gradient_penalty:
                self.generator = Can64Generator(self.z_noise, self.channels, self.num_gen_filters).to(self.device)
                self.discriminator = WCangp64Discriminator(self.channels,self.y_dim, self.num_disc_filters).to(self.device)
                style_criterion = nn.CrossEntropyLoss()


            else:
                self.generator = Can64Generator(self.z_noise, self.channels, self.num_gen_filters).to(self.device)
                self.discriminator =WCangp64Discriminator(self.channels,self.y_dim, self.num_disc_filters).to(self.device)
                style_criterion = nn.CrossEntropyLoss()
        
        # setup optimizers
        # todo : options for SGD, RMSProp
        if self.gradient_penalty:
            self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
            self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.9))
        else:
            self.disc_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)
            self.gen_optimizer = optim.RMSprop(self.generator.parameters(), lr=self.lr)
            

        criterion = nn.BCELoss()
        if self.gradient_penalty:
            self.discriminator.apply(self.weight_init)
            self.generator.apply(self.weight_init) 
        else:
            self.discriminator.apply(self.weights_init)
            self.generator.apply(self.weights_init) 
        if self.disc_path != '':
            self.discriminator.load_state_dict(torch.load(self.disc_path))

        print(str(self.discriminator))
        print(str(self.generator))
        model_file.write("Discriminator:\n")   
        model_file.write(str(self.discriminator))
        model_file.write("\nGenerator:\n")   
        model_file.write(str(self.generator))

        real_label = 1
        fake_label = 0
        lambda_ = 10
        # thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
        one = torch.tensor(1.0)
        mone = torch.tensor(-1.0)  
        one = one.to(self.device)
        mone = mone.to(self.device)


   
        # Normalized noise
        fixed_noise = torch.randn(self.batch_size, self.z_noise, 1, 1,device=self.device)
        label = torch.full((self.batch_size,), real_label, device=self.device)

        # Generator class/style labels
        gen_style_labels = torch.LongTensor(self.batch_size)
        gen_style_labels = gen_style_labels.fill_(1).to(self.device)
        gen_iterations = 0
      
        # Actual training!
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            training_notes_file.write("\nEpoch"+str(epoch)+":\n")
            data_iterator = iter(dataloader)
            i = 0
            wdcgan_flag = False
            # Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py
            while i < len(dataloader):
                j = 0
                disc_loss_epoch = []
                gen_loss_epoch = []
                if self.type == "can":
                    disc_class_loss_epoch = []
                    gen_class_loss_epoch = []
                
                if self.gradient_penalty == False:
                    if gen_iterations < 25 or (gen_iterations % 500 == 0):
                        disc_iters = 100
                else:
                    disc_iters = self.disc_iterations

                while j < disc_iters and (i < len(dataloader)):
                    
                    if self.gradient_penalty == False:
                        # Train Discriminator
                        for param in self.discriminator.parameters():
                            param.data.clamp_(self.lower_clamp,self.upper_clamp)
                    

                    for param in self.discriminator.parameters():
                        param.requires_grad_(True)
                    
                    j+=1
                    i+=1
                    data = data_iterator.next()
                    self.discriminator.zero_grad()
                    real_images, image_labels = data
                    real_images = real_images.to(self.device) 
                    batch_size = real_images.size(0)
                    real_image_labels = torch.LongTensor(batch_size).to(self.device)
                    real_image_labels.copy_(image_labels)
    
                    labels = torch.full((batch_size,),real_label,device=self.device)

                    if self.type == 'can':
                        predicted_output_real, predicted_styles_real = self.discriminator(real_images.detach())
                        predicted_styles_real = predicted_styles_real.to(self.device)
                        disc_class_loss = style_criterion(predicted_styles_real,real_image_labels)
                        disc_class_loss.backward()
                     
                    else:
                        predicted_output_real = self.discriminator(real_images.detach())
                    
                    # disc_loss_real = criterion(predicted_output_real,labels)
                    # disc_loss_real.backward(retain_graph=True)

                    disc_loss_real = torch.mean(predicted_output_real)
                    

                    # fake

                    noise = torch.randn(batch_size,self.z_noise)
                    noise = noise.to(device=self.device)
                    with torch.no_grad():
                        noise_g = noise.detach()
                    fake_images = self.generator(noise_g)
                    labels.fill_(fake_label)

                    if self.type == 'can':
                        predicted_output_fake, predicted_styles_fake = self.discriminator(fake_images)

                    else:
                        predicted_output_fake = self.discriminator(fake_images)

            

                    disc_gen_z_1 = predicted_output_fake.mean().item()

                    disc_loss_fake = torch.mean(predicted_output_fake)

                    # is negative
                    #disc_loss =  disc_loss_real - disc_loss_fake

                    #via https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/WGAN_GP.py
                    if self.gradient_penalty:
                        # gradient penalty
                        gradient_penalty = self.calc_gradient_penalty(self.discriminator,real_images,fake_images)
                        disc_loss  =  disc_loss_fake  - disc_loss_real + gradient_penalty

                    else:
                        disc_loss  =  disc_loss_fake  - disc_loss_real
                    
                
                    if self.type == 'can':
                        disc_loss += disc_class_loss.mean()

                    disc_x = disc_loss.mean().item()
                    disc_loss.backward()
                    self.disc_optimizer.step()
                    


                # train generator
                for param in self.discriminator.parameters():
                    param.requires_grad_(False)

                self.generator.zero_grad()
                noise = torch.randn(batch_size,self.z_noise)
                noise = noise.to(device=self.device)                
                fake_images = self.generator(noise)

                if self.type == 'can':
                    predicted_output_fake, predicted_styles_fake = self.discriminator(fake_images)
                    predicted_styles_fake = predicted_styles_fake.to(self.device)

                else:
                    predicted_output_fake = self.discriminator(fake_images)
                
                # gen_loss = criterion(predicted_output_fake,labels)
                if self.gradient_penalty:
                    gen_loss = torch.mean(predicted_output_fake)
                    gen_loss.backward(mone)
                    gen_loss = -gen_loss
                else:
                    gen_loss = -torch.mean(predicted_output_fake)
                disc_gen_z_2 = gen_loss.mean().item()

                if self.type == 'can':
                    fake_batch_labels = 1.0/self.y_dim * torch.ones_like(predicted_styles_fake)
                    fake_batch_labels = torch.mean(fake_batch_labels,1).long().to(self.device)
                    gen_class_loss = style_criterion(predicted_styles_fake,fake_batch_labels)
                    gen_class_loss.backward()
                    gen_loss += gen_class_loss.mean()
                    # not in the paper
                    #disc_loss -= torch.log(gen_class_loss)
                if not self.gradient_penalty:
                    gen_loss.backward()
                gen_iterations += 1
                self.gen_optimizer.step()

               

                disc_loss_epoch.append(disc_loss.item())
                gen_loss_epoch.append(gen_loss.item())    

                fixed_noise = torch.randn(batch_size,self.z_noise)
                fixed_noise = fixed_noise.to(device=self.device)     
                if self.type=="can":
                    disc_class_loss_epoch.append(disc_class_loss.item())
                    gen_class_loss_epoch.append(gen_class_loss.item())
                    
        
                if self.type=='can':
                

                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Class_D: %.4f Class_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                        % (epoch, self.num_epochs, i, len(dataloader),
                            disc_loss.item(), gen_loss.item(), disc_class_loss.item(), gen_class_loss.item(), disc_x, disc_gen_z_1, disc_gen_z_2))
                    # training_notes_file.write('\n[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Class_D: %.4f Class_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    #     % (epoch, self.num_epochs, i, len(dataloader),
                    #         disc_loss.item(), gen_loss.item(), disc_class_loss.item(), gen_class_loss.item(), disc_x, disc_gen_z_1, disc_gen_z_2))
                else:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f  D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, self.num_epochs, i, len(dataloader),
                        disc_loss.item(), gen_loss.item(),  disc_x, disc_gen_z_1, disc_gen_z_2))
                    # training_notes_file.write('\n[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Class_D: %.4f Class_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    #     % (epoch, self.num_epochs, i, len(dataloader),
                    #         disc_loss.item(), gen_loss.item(), disc_class_loss.item(), gen_class_loss.item(), disc_x, disc_gen_z_1, disc_gen_z_2))            
                if (i % 10000 == 0) or i == (len(dataloader) -1) or wdcgan_flag==False:
                    if gradient_penalty:
                        with torch.no_grad():
    	                    noisev = noise 
                        samples = self.generator(noisev)
                        samples = samples.view(self.batch_size, 3, 64, 64)
                        vutils.save_image(samples.data,
                            '%s/fake_samples_epoch_%03d_%04d.jpg' % (self.out_folder, epoch,i),
                            normalize=True)
                    else:
                        with torch.no_grad():
    	                    noisev = noise 
                        samples = self.generator(noisev)
                        vutils.save_image(samples.data,
                        '%s/fake_samples_epoch_%03d_%04d.jpg' % (self.out_folder, epoch,i),
                        normalize=True)
                    wdcgan_flag=True
             


            epoch_end_time = time.time()
            training_notes_file.write("\nEpoch training time: {} seconds".format(epoch_end_time-epoch_start_time))


             
              # Metrics for Floydhub
            print('{{"metric": "disc_loss", "value": {:.4f}}}'.format(np.mean(disc_loss_epoch)))
            print('{{"metric": "gen_loss", "value": {:.4f}}}'.format(np.mean(gen_loss_epoch)))

            training_notes_file.write("\nEpoch "+str(epoch)+" average losses:\n")
            training_notes_file.write("Discriminator loss: "+ str(np.mean(disc_loss_epoch)))
            training_notes_file.write("\nGenerator loss: "+str(np.mean(gen_loss_epoch)))
            
            if self.type=='can':
                print('{{"metric": "disc_class_loss", "value": {:.4f}}}'.format(np.mean(disc_class_loss_epoch)))
                print('{{"metric": "gen_class_loss", "value": {:.4f}}}'.format(np.mean(gen_class_loss_epoch)))
                training_notes_file.write("\nDiscriminator classification loss: "+str(np.mean(disc_class_loss_epoch)))
                training_notes_file.write("\nGenerator `classification` loss: "+ str(np.mean(gen_class_loss_epoch)))

            # Get the mean of the losses over the epoch for the loss graphs
            disc_loss_epoch = np.asarray(disc_loss_epoch)
            gen_loss_epoch = np.asarray(gen_loss_epoch)
            if self.type == 'can':
                disc_class_loss_epoch = np.asarray(disc_class_loss_epoch)
                self.train_history['disc_class_loss'].append(np.mean(disc_class_loss_epoch))
                gen_class_loss_epoch = np.asarray(gen_class_loss_epoch)
                self.train_history['gen_class_loss'].append(np.mean(gen_class_loss_epoch))


            self.train_history['disc_loss'].append(np.mean(disc_loss_epoch))
            self.train_history['gen_loss'].append(np.mean(gen_loss_epoch))
            
            # do checkpointing
            if (epoch > 1 and (epoch % 5 == 0)) or  (epoch == self.num_epochs -1):
                torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (self.out_folder, epoch))
                torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (self.out_folder, epoch))
                
          
            training_notes_file.write("\n---------------------------------------------------------------------------------\n")

        training_notes_file.write("\nTotal training time: {} seconds".format(time.time()-start_time))
        model_file.close()
        training_notes_file.close()
        losses_file.write(str(self.train_history))
        losses_file.close()