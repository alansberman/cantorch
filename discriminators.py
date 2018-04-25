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



# WGAN discriminator
class WganDiscriminator(nn.Module):
    
    def __init__(self):
        super(WganDiscriminator,self).__init__()


