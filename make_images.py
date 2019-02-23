
import argparse
import os
from generators import *
from discriminators import *
from ops import *
from utils import *
import random
import torch

import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark=True

import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='dcgan')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--z_noise', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--channels',type=int,help="number of image channels",default=3)
parser.add_argument('--num_gen_filters', type=int, default=64)
parser.add_argument('--num_disc_filters', type=int, default=64)
parser.add_argument('--gen_path', default='C:\\Users\\alan\\Desktop\\experiments\\dcgan_imagenet_64\\netG_epoch_24.pth', help="path to netG (to continue training)")
parser.add_argument('--cuda',  action='store_true',default=True, help='enables cuda')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--out_folder', default="C:\\Users\\alan\\Desktop\\experiments\\images", help='folder to output images and model checkpoints') #"/output"

options = parser.parse_args()
folder = options.out_folder + "\\"+options.name
print(folder)
os.system('mkdir {0}'.format(folder))

cudnn.benchmark = True
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if options.name == 'dcgan':
        generator = DcganGenerator(options.z_noise, options.channels, options.num_gen_filters).to(device)
        

    generator.load_state_dict(torch.load(options.gen_path))
    for i in range(1000):
        if options.name=='dcgan':
            noise = torch.randn(options.batch_size, options.z_noise, 1, 1, device=device)
        noise = noise.to(device)
        with torch.no_grad():
            noisev = noise 
            samples = generator(noisev)
            vutils.save_image(samples.data,'%s_%04d.jpg' % (folder+"\\", i),normalize=True)      

if __name__ == '__main__':
    main()
