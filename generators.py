import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
import torchvision.utils as vutils
from torch.autograd import Variable

# Base dcgan generator
class DcganGenerator(nn.Module):
    """

    Finally, convert this high level representation into a 256 × 256 pixel image. In
    other words, starting from z ∈ R
    100 → 4 × 4 × 1024 → 8 × 8 × 1024 → 16 × 16 × 512 →
    32 × 32 × 256 → 64 × 64 × 128 → 128 × 128 × 64 → 256 × 256 × 3 (the generated image size).

    """
    def __init__(self, z_noise, image_size, channels, num_gen_filters, pow=4):
        super(DcganGenerator,self).__init__()

        self.main = nn.Sequential()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Add main layers 
        # TY https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py

        current_gen_filters = num_gen_filters*(2*pow)
        self.main.add_module('initial_conv_transpose.{0}-{1}'.format(z_noise, current_gen_filters), nn.ConvTranspose2d(z_noise, current_gen_filters, 4, 1, 0, bias=False))
        self.main.add_module('initial_batch_norm.{0}'.format(current_gen_filters),nn.BatchNorm2d(current_gen_filters))
        self.main.add_module('initial_relu.{0}'.format(current_gen_filters), nn.LeakyReLU(0.2, inplace=True))

        
        #     num_disc_filters = num_gen_filters
        
        while current_gen_filters > num_gen_filters:
            # Add the next layer
            self.main.add_module('conv_transpose_layer.{0}-{1}'.format(current_gen_filters,current_gen_filters//2),
            nn.ConvTranspose2d(current_gen_filters,current_gen_filters//2, 4, 2, 1, bias=False))
            # Batchnormalize
            self.main.add_module('batch_norm.{0}'.format(current_gen_filters//2),nn.BatchNorm2d(current_gen_filters//2))
            # ReLU Activation
            self.main.add_module('relu.{0}'.format(current_gen_filters//2), nn.LeakyReLU(0.2, inplace=True))
            # Update features
            current_gen_filters = current_gen_filters // 2
            
        
        # Add final layer 
        self.main.add_module('final_layer.{0}-{1}'.format(num_gen_filters, channels), nn.ConvTranspose2d(num_gen_filters, channels, 4, 2, 1, bias=False))
        # Tanh it 
        self.main.add_module('tanh',nn.Tanh())

        # self.main = nn.Sequential(
        #     # inp is Z, going into a convolution
        #     # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        #     # 6 layers, stride 2 , 1 padding
        #     #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
    
        #     # # Layer 1 100 -> 4x4x1024
        #     # nn.ConvTranspose2d( z_noise, num_gen_filters * 16, kernel_size=4, stride=1, padding=0, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 2 -> 8x8x1024
        #     # nn.ConvTranspose2d( num_gen_filters*16, num_gen_filters * 16, kernel_size=4, stride=2, padding=1, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 3 -> 16*16*512
        #     # nn.ConvTranspose2d( num_gen_filters*16, num_gen_filters * 8, kernel_size=4, stride=2, padding=1, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 4 -> 32*32*256
        #     # nn.ConvTranspose2d( num_gen_filters*8, num_gen_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 5 -> 64*64*128
        #     # nn.ConvTranspose2d( num_gen_filters*4, num_gen_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 5 -> 128*128*64
        #     # nn.ConvTranspose2d( num_gen_filters*2, num_gen_filters, kernel_size=4, stride=2, padding=1, bias=False),
        #     # # batch norm??
        #     # nn.ReLU(True),

        #     # # Layer 6 -> 64*64*3
        #     # nn.ConvTranspose2d(num_gen_filters,  channels, kernel_size=4, stride=2, padding=1, bias=False),
        #     # nn.Tanh()

        # self.main = nn.Sequential(
        # nn.ConvTranspose2d( z_noise, num_gen_filters * 8, 4, 1, 0, bias=False),
        # nn.BatchNorm2d(num_gen_filters * 8),
        # nn.ReLU(True),
        # # state size. (num_gen_filters*8) x 4 x 4
        # nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(num_gen_filters * 4),
        # nn.ReLU(True),
        # # state size. (num_gen_filters*4) x 8 x 8
        # nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(num_gen_filters * 2),
        # nn.ReLU(True),
        # # state size. (num_gen_filters*2) x 16 x 16
        # nn.ConvTranspose2d(num_gen_filters * 2,     num_gen_filters, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(num_gen_filters),
        # nn.ReLU(True),
        # # state size. (num_gen_filters) x 32 x 32
        # nn.ConvTranspose2d(    num_gen_filters,  channels, 4, 2, 1, bias=False),
        # nn.Tanh()
        #     # state size. (channels) x 64 x 64
        # )
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        
        return output


# CAN generator
class CanGenerator(nn.Module):
    
    def __init__(self):
        super(CanGenerator,self).__init__()


# WGAN generator
class WganGenerator(nn.Module):
    
    def __init__(self):
        super(WganGenerator,self).__init__()