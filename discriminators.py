import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
# Base dcgan discriminator
class DcganDiscriminator(nn.Module):
  
    def __init__(self, image_size, channels, num_disc_filters, power=4):
        super(DcganDiscriminator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(

            nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (num_disc_filters*4) x 8 x 8
            # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(num_disc_filters * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_disc_filters * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # ty  https://github.com/pytorch/examples/issues/70 apaske
        )
    
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        
        return output.view(-1, 1).squeeze(1)


# CAN discriminator
class CanDiscriminator(nn.Module):
    
    def __init__(self,image_size, channels,y_dim, num_disc_filters,power=4):
        super(CanDiscriminator, self).__init__()
        self.ngpu = 1
        self.conv = nn.Sequential(
                nn.Conv2d(channels, num_disc_filters//2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
          
                nn.Conv2d(num_disc_filters//2, num_disc_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(num_disc_filters),
                nn.LeakyReLU(0.2, inplace=True),
         
                nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(num_disc_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
           
                nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(num_disc_filters * 4),
                nn.LeakyReLU(0.2, inplace=True),
    
                nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(num_disc_filters * 8),
                nn.LeakyReLU(0.2, inplace=True),
    
            )
        self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
        
        self.real_fake_head = nn.Linear(512*4*4, 1)
        
        self.sig = nn.Sigmoid()
        self.fc = nn.Sequential() 
        self.fc.add_module("linear_layer.{0}".format(num_disc_filters*16),nn.Linear(512*4*4,num_disc_filters*16))
        self.fc.add_module('relu.{0}'.format(num_disc_filters*16), nn.LeakyReLU(0.2, inplace=True))
        self.fc.add_module("linear_layer.{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
        self.fc.add_module('relu.{0}'.format(num_disc_filters*8), nn.LeakyReLU(0.2, inplace=True))
        self.fc.add_module("linear_layer.{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
        self.fc.add_module('relu.{0}'.format(num_disc_filters), nn.LeakyReLU(0.2, inplace=True))
        self.fc.add_module('softmax',nn.Softmax(dim=1))
       
    def forward(self, inp):

        x = self.conv(inp)
        x = self.final_conv(x) 
        x = x.view(x.size(0),-1) 

        real_out = self.sig(self.real_fake_head(x))
        real_out = real_out.view(-1,1).squeeze(1)
        style = self.fc(x) 
        #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
        return real_out,style


class Can64Discriminator(nn.Module):
        
    def __init__(self,image_size, channels,y_dim, num_disc_filters,power=4):
            super(Can64Discriminator, self).__init__()
            self.ngpu = 1
            self.conv = nn.Sequential(
                    nn.Conv2d(channels, num_disc_filters//2, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters//2, num_disc_filters, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
        
                    nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
        
                )
            self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
            
            self.real_fake_head = nn.Linear(num_disc_filters * 8, 1)
            
            self.sig = nn.Sigmoid()
            self.fc = nn.Sequential() 
            self.fc.add_module("linear_layer.{0}".format(num_disc_filters*16),nn.Linear(num_disc_filters*8,num_disc_filters*16))
            self.fc.add_module('relu.{0}'.format(num_disc_filters*16), nn.LeakyReLU(0.2, inplace=True))
            self.fc.add_module("linear_layer.{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
            self.fc.add_module('relu.{0}'.format(num_disc_filters*8), nn.LeakyReLU(0.2, inplace=True))
            self.fc.add_module("linear_layer.{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
            self.fc.add_module('relu.{0}'.format(num_disc_filters), nn.LeakyReLU(0.2, inplace=True))
            self.fc.add_module('softmax',nn.Softmax(dim=1))
        
    def forward(self, inp):

        x = self.conv(inp)
        x = self.final_conv(x) 
        x = x.view(x.size(0),-1) 
        real_out = self.sig(self.real_fake_head(x))
        real_out = real_out.view(-1,1).squeeze(1)
        style = self.fc(x) 
        #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
        return real_out,style



# WGAN discriminator
class WganDiscriminator(nn.Module):
    
    def __init__(self):
        super(WganDiscriminator,self).__init__()

   # minimizing −Ex∼pdata [log Dr(x) + log Dc(c = ˆc|x)] for the real images and −Ez∼pz[log(1 − Dr(G(z)))] for the generated images
    # generator
    # maximizing log(1 −Dr(G(z)) − sigmaK k=1((1/K)log(Dc(ck|G(z)) + (1 − (1/K)log(1 − Dc(ck|G(z)).

    

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
