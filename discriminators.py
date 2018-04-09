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

#Base dcgan discriminator
class _dcganDiscriminator(nn.Module):
    
    def __init__(self):
        super(_dcganDiscriminator,self).__init__()
