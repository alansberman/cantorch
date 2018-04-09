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

# Base dcgan generator
class DcganGenerator(nn.Module):
    
    def __init__(self, z_noise, channels, num_gen_filters):
        super(DcganGenerator,self).__init__()
        self.main = nn.Sequential(
            # inp is Z, going into a convolution
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 

            nn.ConvTranspose2d(z_noise, num_gen_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_gen_filters * 8),
            nn.ReLU(True),
            # state size. (num_gen_filters*8) x 4 x 4
            nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 4),
            nn.ReLU(True),
            # state size. (num_gen_filters*4) x 8 x 8
            nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 2),
            nn.ReLU(True),
            # state size. (num_gen_filters*2) x 16 x 16
            nn.ConvTranspose2d(num_gen_filters * 2, num_gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters),
            nn.ReLU(True),
            # state size. (num_gen_filters) x 32 x 32
            nn.ConvTranspose2d(num_gen_filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )
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