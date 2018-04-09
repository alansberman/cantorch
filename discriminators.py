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

# Base dcgan discriminator
class DcganDiscriminator(nn.Module):
    
    def __init__(self, channels, num_disc_filters):
        super(DcganDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # inp is (channels) x 64 x 64
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
            nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters) x 32 x 32
            nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*2) x 16 x 16
            nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*4) x 8 x 8
            nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_disc_filters*8) x 4 x 4
            # 2nd arg was 1
            nn.Conv2d(num_disc_filters * 8, num_disc_filters * 16 , 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_disc_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_filters * 16, 1 , 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)



# CAN discriminator
class CanDiscriminator(nn.Module):
    
    def __init__(self):
        super(CanDiscriminator,self).__init__()


# WGAN discriminator
class WganDiscriminator(nn.Module):
    
    def __init__(self):
        super(WganDiscriminator,self).__init__()
