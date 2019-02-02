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

    def __init__(self, z_noise, channels, num_gen_filters):
        super(DcganGenerator,self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            # was num_gen_filters * 16
            nn.ConvTranspose2d(z_noise, (num_gen_filters * 8 ), 4, 1, 0, bias=False),
            nn.BatchNorm2d((num_gen_filters * 8 )),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_gen_filters * 8, num_gen_filters * 4, 4, 2, 1, bias=False),
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
    def __init__(self, z_noise, channels, num_gen_filters):
        super(Can64Generator,self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
        nn.ConvTranspose2d(z_noise, num_gen_filters * 16, 4, 1, 0, bias=False),
        nn.BatchNorm2d(num_gen_filters * 16),
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
        nn.ConvTranspose2d(num_gen_filters, 3, 4, 2, 1, bias=False),
        nn.Tanh()
        )
    def forward(self, inp):
        output = self.main(inp)
        return output


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

DIM = 64
OUTPUT_DIM = 64*64*4
# thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output
# thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

# and thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

# thanks to # and https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output



#with thanks to https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
# and https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py
class GoodGenerator(nn.Module):
    def __init__(self, dim=DIM,output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4*4*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample = 'up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample = 'up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, OUTPUT_DIM)
        return output

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
        output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, 4*4*8*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output