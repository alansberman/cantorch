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

class WGANGPDiscriminator(nn.Module):
  
    def __init__(self, channels, num_disc_filters):
        super(WGANGPDiscriminator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(

            nn.Conv2d(channels, num_disc_filters, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 3, 2, 1, bias=False),
            #nn.LayerNorm(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 3, 2, 1, bias=False),
            #nn.LayerNorm(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (num_disc_filters*4) x 8 x 8
            # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(num_disc_filters * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            # was num_disc_filters * 16
            # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            # #nn.LayerNorm(num_disc_filters * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # # was num_disc_filters * 16
            # nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
            # # ty  https://github.com/pytorch/examples/issues/70 apaske
        )
        self.linear = nn.Linear(4*4*4*num_disc_filters,1)
    
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
            output = output.view(-1,4*4*4*64)
            output = self.linear(output)
        return output
        
        # return output.view(-1, 1).squeeze(1)


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



class DiscriminatorOrig(nn.Module):
    def __init__(self, ngpu,channels,num_disc_filters):
        super(DiscriminatorOrig, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



# class DDiscriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(DDiscriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(num_disc_filters * 4, num_disc_filters* 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)

#         return output.view(-1, 1).squeeze(1)




# class Generator3(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator3, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(     nz, ngf * 4, 4, 4, 0, bias=False),     # def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, w5, g5, b5, w6, g6, b6, wx):
#                                                                             #     h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
#                                                                             #     h = h.reshape((h.shape[0], ngf*4, 4, 4))
#                                                                             #     h2 = relu(batchnorm(deconv(h, (512, 512, 3, 3), subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
#                                                                             #     h3 = relu(batchnorm(deconv(h2, (512, 256, 3, 3), subsample=(1, 1), border_mode=(1, 1)), g=g3, b=b3))
#                                                                             #     h4 = relu(batchnorm(deconv(h3, (256, 256, 3, 3), subsample=(2, 2), border_mode=(1, 1)), g=g4, b=b4))
#                                                                             #     h5 = relu(batchnorm(deconv(h4,  (256, 128, 3, 3), subsample=(1, 1), border_mode=(1, 1)), g=g5, b=b5))
#                                                                             #     h6 = relu(batchnorm(deconv(h5, (128, 3, 3, 3), subsample=(2, 2), border_mode=(1, 1)), g=g6, b=b6))
#                                                                             #     x = tanh(deconv(h6, (128, 3, 3, 3), subsample=(1, 1), border_mode=(1, 1)))
#                                                                             #     return x

#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2 , 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2 ),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             # was 4
#             nn.ConvTranspose2d(ngf * 2,  ngf, 1, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    ngf ,      nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         #
#         return output


# class Discriminator2(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator2, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(num_disc_filters, num_disc_filters, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(num_disc_filters , num_disc_filters * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(num_disc_filters * 2, num_disc_filters * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(num_disc_filters * 4, num_disc_filters * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(num_disc_filters * 4, 1, 2, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)

#         return output.view(-1, 1).squeeze(1)


# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),

#             # HAD TO REMOVE FOR 32x32
#             # nn.ConvTranspose2d(ngf * 2,     ngf, 1, 2, 1, bias=False),
#             # nn.BatchNorm2d(ngf),
#             # nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    ngf * 2,      nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         #
#         return output        


# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_disc_filters * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(num_disc_filters * 8, 1, 2, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)

#         return output.view(-1, 1).squeeze(1)



# # custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)




class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(channels, num_disc_filters, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(num_disc_filters * 8, num_disc_filters * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(num_disc_filters * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



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


class Discriminator32(nn.Module):
    def __init__(self, ngpu,channels,num_disc_filters): 
        super(Discriminator32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_disc_filters * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


#thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class GoodDiscriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim

        self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=DIM)
        self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(DIM/2))
        self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/4))
        self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/8))
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)

    def forward(self, input):
        output = input.contiguous()
        #32,32 was DIM,DIM
        output = output.view(-1, 3, 32, 32)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output
