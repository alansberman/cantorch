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
  
    def __init__(self, channels, num_disc_filters):
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

            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
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
        
    def __init__(self, channels,y_dim, num_disc_filters):
            super(Can64Discriminator, self).__init__()
            self.ngpu = 1
            self.conv = nn.Sequential(
                    nn.Conv2d(channels, num_disc_filters // 2, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters // 2, num_disc_filters, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
        
                    nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
        
                )
            # was this
            #self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
            
            self.real_fake_head = nn.Linear(num_disc_filters * 8, 1)
            
            # no bn and lrelu needed
            self.sig = nn.Sigmoid()
            self.fc = nn.Sequential() 
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*16),nn.Linear(num_disc_filters*8,num_disc_filters*16))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
            self.fc.add_module('softmax',nn.Softmax(dim=1))
        
    def forward(self, inp):
        x = self.conv(inp)
        x = x.view(x.size(0),-1) 
        real_out = self.sig(self.real_fake_head(x))
        real_out = real_out.view(-1,1).squeeze(1)
        style = self.fc(x) 
        #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
        return real_out,style


class WCan64Discriminator(nn.Module):
        
    def __init__(self, channels,y_dim, num_disc_filters):
            super(WCan64Discriminator, self).__init__()
            self.ngpu = 1
            self.conv = nn.Sequential(
                    nn.Conv2d(channels, num_disc_filters // 2, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters // 2, num_disc_filters, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
        
                    nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(num_disc_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
        
                )
            # was this
            #self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
            
            self.real_fake_head = nn.Linear(num_disc_filters * 8, 1)
            
            # no bn and lrelu needed
            self.fc = nn.Sequential() 
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*16),nn.Linear(num_disc_filters*8,num_disc_filters*16))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
            self.fc.add_module('softmax',nn.Softmax(dim=1))
        
    def forward(self, inp):
        x = self.conv(inp)
        x = x.view(x.size(0),-1) 
        real_out = self.real_fake_head(x)
        real_out = real_out.view(-1,1).squeeze(1)
        style = self.fc(x) 
        #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
        return real_out,style


# Base dcgan discriminator
class WganDiscriminator(nn.Module):
  
    def __init__(self, channels, num_disc_filters):
        super(WganDiscriminator, self).__init__()
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

            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
            # ty  https://github.com/pytorch/examples/issues/70 apaske
        )
    
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        
        return output.view(-1, 1).squeeze(1)


# Layernorm
class WgangpDiscriminator(nn.Module):
  
    def __init__(self, channels, num_disc_filters):
        super(WgangpDiscriminator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(

            nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            #nn.LayerNorm(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            #nn.LayerNorm(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (num_disc_filters*4) x 8 x 8
            # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(num_disc_filters * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            #nn.LayerNorm(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # was num_disc_filters * 16
            nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
            # ty  https://github.com/pytorch/examples/issues/70 apaske
        )
    
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        
        return output.view(-1, 1).squeeze(1)

# class WCangp64Discriminator(nn.Module):
        
#     def __init__(self, channels,y_dim, num_disc_filters):
#             super(WCangp64Discriminator, self).__init__()
#             self.ngpu = 1
#             self.conv1 = nn.Conv2d(channels, num_disc_filters // 2, 4, 2, 1, bias=False)
#             self.lr1 = nn.LeakyReLU(0.2, inplace=True)
            
#             self.conv2 = nn.Conv2d(num_disc_filters // 2, num_disc_filters, 4, 2, 1, bias=False)
#             self.ln1 = nn.LayerNorm(self.conv2.get_shape()[1:])
#             self.lr2 = nn.LeakyReLU(0.2, inplace=True)
            
#             self.conv3 = nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False)
#             self.ln2 = nn.LayerNorm(self.conv3.get_shape()[1:])
#             self.lr3 = nn.LeakyReLU(0.2, inplace=True)
            
#             self.conv4 = nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False)
#             self.ln3 = nn.LayerNorm(self.conv4.get_shape()[1:])
#             self.lr4 = nn.LeakyReLU(0.2, inplace=True)
        
#             self.conv5 = nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 1, 0, bias=False)
#             self.ln4 = nn.LayerNorm(self.conv5.get_shape()[1:])
#             self.lr5 = nn.LeakyReLU(0.2, inplace=True)
        
#             # was this
#             #self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
            
#             self.real_fake_head = nn.Linear(num_disc_filters * 8, 1)
            
#             # no bn and lrelu needed
#             self.fc = nn.Sequential() 
#             self.fc.add_module("linear_layer{0}".format(num_disc_filters*16),nn.Linear(num_disc_filters*8,num_disc_filters*16))
#             self.fc.add_module("linear_layer{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
#             self.fc.add_module("linear_layer{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
#             self.fc.add_module('softmax',nn.Softmax(dim=1))
        
#     def forward(self, inp):
#         x = self.lr1(self.conv1(x))
#         x = self.lr2(self.ln1(self.conv2(x)))
#         x = self.lr3(self.ln2(self.conv3(x)))
#         x = self.lr4(self.ln3(self.conv4(x)))
#         x = self.lr5(self.ln4(self.conv5(x)))
#         x = x.view(x.size(0),-1) 
#         real_out = self.real_fake_head(x)
#         real_out = real_out.view(-1,1).squeeze(1)
#         style = self.fc(x) 
#         #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
#         return real_out,style


class WCangp64Discriminator(nn.Module):
        
    def __init__(self, channels,y_dim, num_disc_filters):
            super(WCangp64Discriminator, self).__init__()
            self.ngpu = 1
            self.conv = nn.Sequential(
                    nn.Conv2d(channels, num_disc_filters // 2, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters // 2, num_disc_filters, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
            
                    nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
        
                    nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 1, 0, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
        
                )
            # was this
            #self.final_conv = nn.Conv2d(num_disc_filters * 8, num_disc_filters * 8, 4, 2, 1, bias=False)
            
            self.real_fake_head = nn.Linear(num_disc_filters * 8, 1)
            
            # no bn and lrelu needed
            self.fc = nn.Sequential() 
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*16),nn.Linear(num_disc_filters*8,num_disc_filters*16))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters*8),nn.Linear(num_disc_filters*16,num_disc_filters*8))
            self.fc.add_module("linear_layer{0}".format(num_disc_filters),nn.Linear(num_disc_filters*8,y_dim))
            self.fc.add_module('softmax',nn.Softmax(dim=1))
        
    def forward(self, inp):
        x = self.conv(inp)
        x = x.view(x.size(0),-1) 
        real_out = self.real_fake_head(x)
        real_out = real_out.view(-1,1).squeeze(1)
        style = self.fc(x) 
        #style = torch.mean(style,1) # CrossEntropyLoss requires input be (N,C)
        return real_out,style