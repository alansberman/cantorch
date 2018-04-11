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
    """

As described earlier, the discriminator has two losses (real/fake loss and multi-label loss). The
discriminator in our work starts by a common body of convolution layers followed by two heads
(one for the real/fake loss and one for the multi-label loss). The common body of convolution layers
is composed of a series of six convolution layers (all with stride 2 and 1 pixel padding). conv1 (32
4 × 4 filters), conv2 (64 4 × 4 filters, conv3 (128 4 × 4 filters, conv4 (256 4 × 4 filters, conv5 (512
4 × 4 filters, conv6 (512 4 × 4 filters). Each convolutional layer is followed by a leaky rectified
activation (LeakyRelU) [13, 25] in all the layers of the discriminator. After passing a image to
the common conv D body, it will produce a feature map or size (4 × 4 × 512). The real/fake Dr
head collapses the (4 × 4 × 512) by a fully connected to produce Dr(c|x) (probability of image
coming for the real image distribution). The multi-label probabilities Dc(ck|x) head is produced
by passing the(4 × 4 × 512) into 3 fully collected






    batchnorm2d epsilon=1e-5, momentum = 0.9
  Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    """


    def __init__(self, image_size, channels, num_disc_filters, num_extra_layers=0):
        super(DcganDiscriminator, self).__init__()
        self.main = nn.Sequential()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Add main layers 
        # TY https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py
        self.main.add_module('initial_conv.{0}-{1}'.format(channels, num_disc_filters), nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False))
        self.main.add_module('initial_relu.{0}'.format(num_disc_filters), nn.LeakyReLU(0.2, inplace=True))


        filter_size = image_size // 2
        current_disc_filters = num_disc_filters
        
        while filter_size > 4:
            # Set num of input and output features
            input_features, output_features = current_disc_filters, current_disc_filters * 2
            # Add the next layer
            self.main.add_module('conv_layer.{0}-{1}'.format(input_features,output_features),nn.Conv2d(input_features,output_features, 4, 2, 1, bias=False))
            # Batchnormalize
            self.main.add_module('batch_norm.{0}'.format(output_features),nn.BatchNorm2d(output_features))
            # ReLU Activation
            self.main.add_module('relu.{0}'.format(output_features), nn.LeakyReLU(0.2, inplace=True))
            # Update features
            current_disc_filters *= 2
            filter_size = filter_size // 2
        
        # Add final layer 
        self.main.add_module('final_layer.{0}-{1}'.format(num_disc_filters, 1), nn.Conv2d(num_disc_filters, 1, 4, 2, 1, bias=False))
        # Sigmoid it 
        self.main.add_module('sigmoid',nn.Sigmoid())

        
        # self.main = nn.Sequential(
        #     # inp is (channels) x 64 x 64
        #     # isize (64*64), nz, nc, ndf (64), ngpu, n_extra_layers=0

        #     # 6 layers, stride 2 , 1 padding
        #     # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True

        #     # layer 1 conv1 (32 4 × 4 filters)
        #     nn.Conv2d(channels, num_disc_filters//2 , kernel_size=4, stride=2, padding=1, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # layer 2 (64 4 × 4 filters)
        #     nn.Conv2d(num_disc_filters//2, num_disc_filters , kernel_size=4, stride=2, padding=1, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # layer 3 (128 4 × 4 filters)
        #     nn.Conv2d(num_disc_filters, num_disc_filters*2 , kernel_size=4, stride=2, padding=1, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # layer 4  (256 4 × 4 filters)
        #     nn.Conv2d(num_disc_filters*2, num_disc_filters*4 , kernel_size=4, stride=2, padding=1, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # layer 5 (512 4 × 4 filters)
        #     nn.Conv2d(num_disc_filters*4, num_disc_filters*8 , kernel_size=4, stride=2, padding=1, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # layer 6 (512 4 × 4 filters) # second arg orig. ndf*8
        #     # stride HAS TO BE 2 and padding HAS TO BE 1
        #     nn.Conv2d(num_disc_filters*8, 1 , kernel_size=4, stride=1, padding=0, bias=False),
        #     # batch norm ??
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Sigmoid()


        #     # pytorchdcgan
        #     # nn.Conv2d(channels, num_disc_filters, 4, 2, 1, bias=False),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # state size. (num_disc_filters) x 32 x 32
        #     # nn.Conv2d(num_disc_filters, num_disc_filters * 2, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(num_disc_filters * 2),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # state size. (num_disc_filters*2) x 16 x 16
        #     # nn.Conv2d(num_disc_filters * 2, num_disc_filters * 4, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(num_disc_filters * 4),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # state size. (num_disc_filters*4) x 8 x 8
        #     # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(num_disc_filters * 8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # state size. (num_disc_filters*8) x 4 x 4
        #     # nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),
        #     # nn.Sigmoid()


        #     # nn.Conv2d(num_disc_filters * 8, num_disc_filters * 16 , 4, 1, 0, bias=False),
        #     # #nn.Conv2d(num_disc_filters * 8, 1 , 4, 1, 0, bias=False),

        #     # nn.BatchNorm2d(num_disc_filters * 16),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # gotta chagne the 4 to a 1 ?!?!!?
        #     # nn.Conv2d(num_disc_filters * 16, 1 , 4, 1, 0, bias=False),
        #     # nn.Sigmoid()

        #     # nn.Conv2d(num_disc_filters * 4, num_disc_filters * 8, 4, 2, 1, bias=False),
        #     # nn.BatchNorm2d(num_disc_filters * 8),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # # state size. (num_disc_filters*8) x 4 x 4
        #     # nn.Conv2d(num_disc_filters * 8, 1, 4, 1, 0, bias=False),

        # )
    
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
