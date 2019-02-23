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

import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time


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
        self.lsgan = options.lsgan
        self.smoothing = options.smoothing
        self.flip = options.flip
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

        elif self.dataset == 'lsun':
            transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = dset.ImageFolder(root=self.dataroot,transform=transform)  


    # dataset = dset.LSUN(opt.dataroot, classes=['bedroom_train'],
    #                     transform=transforms.Compose([
    #                         transforms.Resize(opt.imageSize),
    #                         transforms.CenterCrop(opt.imageSize),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                     ]))
           

        else:
            # Get dataset
            data = get_dataset(self.dataroot)
       
        # # Set the type of GAN
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                         shuffle=True, num_workers=int(self.workers))

        # # Set the type of GAN
        if self.type == "dcgan": 
            if self.image_size == 32:
                self.generator = Generator32(self.z_noise, self.channels, self.num_gen_filters).to(self.device)
                self.discriminator = Discriminator32(self.ngpu, self.channels, self.num_disc_filters).to(self.device)
            else:
                
                self.discriminator = DcganDiscriminator(self.channels, self.num_disc_filters).to(self.device)
            criterion = nn.BCELoss()

        if self.type == "can":
            self.generator = Can64Generator(self.z_noise, self.channels, self.num_gen_filters).to(self.device)
            self.discriminator = Can64Discriminator(self.channels,self.y_dim, self.num_disc_filters).to(self.device)
            style_criterion = nn.CrossEntropyLoss()
        
        # setup optimizers
        # todo : options for SGD, RMSProp
        if self.smoothing:
            self.disc_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.lr)# betas=(self.beta1, 0.999))
        else:
            self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)# betas=(self.beta1, 0.999))
        # recommended in GANhacks
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)#, betas=(self.beta1, 0.999))
        
        if self.lsgan:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        self.discriminator.apply(self.weights_init)
        self.generator.apply(self.weights_init) 

        if self.disc_path != '':
            self.discriminator.load_state_dict(torch.load(self.disc_path))

        model_file.write("Discriminator:\n")   
        model_file.write(str(self.discriminator))
        model_file.write("\nGenerator:\n")   
        model_file.write(str(self.generator))

        real_label = 1
        fake_label = 0
       
        # Normalized noise
        fixed_noise = torch.randn(self.batch_size, self.z_noise, 1, 1,device=self.device)
      

        # Generator class/style labels
        gen_style_labels = torch.new_ones(self.batch_size,device=self.device)
 

        # Actual training!
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            training_notes_file.write("\nEpoch"+str(epoch)+":\n")
            data_iterator = iter(dataloader)

            i = 0
            # Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py
            while i < len(dataloader):
            
                disc_loss_epoch = []
                gen_loss_epoch = []
                
                if self.type == "can":
                    disc_class_loss_epoch = []
                    gen_class_loss_epoch = []


                curr_start = time.time()
                # WGAN
            
                # Train Discriminator
                self.discriminator.zero_grad()

                # real
                data = data_iterator.next()
                real_images, image_labels = data
                real_images = real_images.to(self.device) 
                batch_size = real_images.size(0)
                real_image_labels = torch.LongTensor(batch_size).to(self.device)
                real_image_labels.copy_(image_labels)

                # label smoothing
                # rand_labels = np.random.uniform(low=0.7, high=1.2, size=(batch_size,))
                # r_labels = torch.from_numpy(rand_labels)
                # labels.copy_(r_labels)
                #print(labels)
                if self.type == 'can':
                    predicted_output_real, predicted_styles_real = self.discriminator(real_images.detach())
                    disc_class_loss = style_criterion(predicted_styles_real,real_image_labels)
                    disc_class_loss.backward(retain_graph=True)
                else:
                    predicted_output_real = self.discriminator(real_images.detach())

                if self.smoothing:
                    labels_real = []
                    labels_fake = []
                    for n in range(self.batch_size):
                        labels_real.append(random.uniform(0.7,1.3))
                        labels_fake.append(random.uniform(0.0,0.3))
                    labels_real = np.asarray(labels_real)
                    labels_fake = np.asarray(labels_fake)
                    if self.flip:
                        prob = random.uniform(0.0,2.0)
                        if prob < 0.3:
                            labels = torch.new_tensor(labels_fake,device=self.device)
                    else:
                        labels = torch.new_tensor(labels_real,device=self.device)
            #labels= torch.full((self.batch_size,), labels_, device=self.device)

                else:
                    if self.flip:
                        prob = random.uniform(0.0,2.0)
                        if prob < 0.3:
                            labels = torch.new_tensor(fake_label,device=self.device)
                    else:
                        labels = torch.full((self.batch_size,), real_label, device=self.device)
                
                disc_loss_real = criterion(predicted_output_real,labels)
                disc_loss_real.backward(retain_graph=True)
                disc_x = predicted_output_real.mean().item()

                # fake

                noise = torch.randn(batch_size,self.z_noise,1,1,device=self.device)
                
                fake_images = self.generator(noise)
                if self.flip:
                    prob = random.uniform(0.0,2.0)
                    if prob < 0.3:
                        if self.smoothing:
                            labels = torch.new_tensor(labels_real)
                        else:
                            labels.fill_(real_label)
                else:
                    labels.fill_(fake_label)
        
                if self.type == 'can':
                    predicted_output_fake, predicted_styles_fake = self.discriminator(fake_images.detach())

                else:
                    predicted_output_fake = self.discriminator(fake_images.detach())
                
                disc_loss_fake = criterion(predicted_output_fake,labels)
                disc_loss_fake.backward(retain_graph=True)
                disc_gen_z_1 = predicted_output_fake.mean().item()
                disc_loss = disc_loss_real + disc_loss_fake

                if self.type == 'can':
                    disc_loss += disc_class_loss 
                
                self.disc_optimizer.step()

                # train generator
         
                self.generator.zero_grad()
                labels.fill_(real_label)

                if self.type == 'can':
                    predicted_output_fake, predicted_styles_fake = self.discriminator(fake_images)

                else:
                    predicted_output_fake = self.discriminator(fake_images)

                gen_loss = criterion(predicted_output_fake,labels)
                gen_loss.backward(retain_graph=True)
                disc_gen_z_2 = predicted_output_fake.mean().item()

                if self.type == 'can':
                    fake_batch_labels = 1.0/self.y_dim * torch.ones_like(predicted_styles_fake)
                    fake_batch_labels = torch.mean(fake_batch_labels,1).long().to(self.device)
                    gen_class_loss = style_criterion(predicted_styles_fake,fake_batch_labels)
                    gen_class_loss.backward()
                    gen_loss += gen_class_loss
                    #disc_loss += torch.log(gen_class_loss)


                self.gen_optimizer.step()     

                disc_loss_epoch.append(disc_loss.item())
                gen_loss_epoch.append(gen_loss.item())    

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
                if (i > 0 and (i % 10000 == 0)) or i == (len(dataloader) -1):
                    fake = self.generator(fixed_noise)
                    vutils.save_image(fake.data,
                            '%s/fake_samples_epoch_%03d_%04d.jpg' % (self.out_folder, epoch,i),
                            normalize=True)
                i += 1

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
        #get_loss_graphs(self.train_history,self.out_folder,self.gan_type)
        #if self.type == 'can':
        #    get_class_loss_graph(self.train_history,self.out_folder,self.gan_type)