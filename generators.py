import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import torchvision.utils as vutils
from torch.autograd import Variable

# Base dcgan generator
class DcganGenerator(nn.Module):

    def __init__(self, z_noise, image_size, channels, num_gen_filters, power=4):
        super(DcganGenerator,self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(

            nn.ConvTranspose2d(z_noise, (num_gen_filters * 16 ), 4, 1, 0, bias=False),
            nn.BatchNorm2d((num_gen_filters * 16 )),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_gen_filters * 16, num_gen_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_gen_filters * 2, num_gen_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_filters),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_gen_filters,channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)
        
        return output



# CAN generator
class CanGenerator(nn.Module):
    
    def __init__(self, z_noise, image_size, channels, num_gen_filters, power=4):
        super(CanGenerator,self).__init__()
        self.ngpu = 1
   
        self.conv1 = nn.ConvTranspose2d(z_noise, 2048, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
      
    def forward(self, inp):
        # if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        # else:
        #     output = self.main(inp)
        
        # return output
        x = inp
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.tanh(self.conv7(x))
        return x


class Can64Generator(nn.Module):
    def __init__(self, z_noise, image_size, channels, num_gen_filters, power=4):
        super(Can64Generator,self).__init__()
        self.ngpu = 1
   
        self.conv1 = nn.ConvTranspose2d(z_noise, num_gen_filters * 16, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_gen_filters * 16)
        self.conv2 = nn.ConvTranspose2d(num_gen_filters * 16, num_gen_filters * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_gen_filters * 4)
        self.conv3 = nn.ConvTranspose2d(num_gen_filters * 4, num_gen_filters * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_gen_filters * 2)
        self.conv4 = nn.ConvTranspose2d(num_gen_filters * 2, num_gen_filters, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_gen_filters)
        # self.conv5 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        # self.bn5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        # self.bn6 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(num_gen_filters, 3, 4, 2, 1, bias=False)
      
    def forward(self, inp):
        # if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        # else:
        #     output = self.main(inp)
        
        # return output
        x = inp
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = F.relu(self.bn6(self.conv6(x)))
        x = F.tanh(self.conv5(x))
        return x


# sigmaK k=1((1/K)log(Dc(ck|G(z)) + (1 − (1/K)log(1 − Dc(ck|G(z)).

class CanGLoss(nn.Module):
    def __init__(self,y_dim,labels,disc_class_layer):
        super(CanGLoss,self).__init__()
        self.y_dim = y_dim
        self.labels = labels
        self.disc_class_layer = disc_class_layer
    def forward(self,inp):
        style_loss = 0
        for i in range(1,self.y_dim+1):
            style_loss += (1/i)*torch.log(self.disc_class_layer(inp)) + (1 - (1/i))*torch.log(1-self.disc_class_layer(inp))
        return style_loss*-1

# WGAN generator
class WganGenerator(nn.Module):
    
    def __init__(self):
        super(WganGenerator,self).__init__()

