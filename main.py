
from model import *
from wasserstein_model import *
import argparse
import os
import random
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

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  help='cifar10 | lsun | imagenet | wikiart',default='imagenet')
parser.add_argument('--dataroot', help='path to dataset',default="/mydata") # D:\\WikiArt\\wikiart\\ #/mydata
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--gan_type', type=str, help='dcgan | wgan | can', default='dcgan')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--optimizer', type=str, default="Adam", help='Adam | SGD | RMSProp')
parser.add_argument('--z_noise', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--y_dim',type=int,help="number of output/target classes",default=10)
parser.add_argument('--channels',type=int,help="number of image channels",default=3)
parser.add_argument('--num_gen_filters', type=int, default=64)
parser.add_argument('--power', type=int, default=8, help="1-number of hidden layers")
parser.add_argument('--disc_iterations', type=int, default=5,help="ratio of discriminator updates to generator")
parser.add_argument('--num_disc_filters', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5') 
parser.add_argument('--lower_clamp', type=float, default=-0.01, help='lower clamp for params')
parser.add_argument('--upper_clamp', type=float, default=0.01, help='upper clamp for params')
parser.add_argument('--cuda',  action='store_true',default=True, help='enables cuda')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gen_path', default='', help="path to netG (to continue training)")
parser.add_argument('--disc_path', default='', help="path to netD (to continue training)")
parser.add_argument('--out_folder', default="/output", help='folder to output images and model checkpoints') #"/output"
parser.add_argument('--wgan', type=bool, default=True, help='if training a la WGAN') #"/output"
parser.add_argument('--lsgan', type=bool, default=False, help='if training with LSGAN') #"/output"
parser.add_argument('--num_critic', type=int, default=5,help="D:G training") #"/output"
parser.add_argument('--gradient_penalty', type=bool, default=True,help="if training with WGANGP") #"/output"


parser.add_argument('--manual_seed', type=int, help='manual seed')

options = parser.parse_args()
if options.out_folder is None:
    options.out_folder = "samples"
    os.system('mkdir {0}'.format(options.out_folder))


if options.manual_seed is None:
    options.manual_seed = random.randint(1, 10000)
print("Random Seed: ", options.manual_seed)
print("Using Cuda? (True/False)",options.cuda)
random.seed(options.manual_seed)
torch.manual_seed(options.manual_seed)
if options.cuda:
    torch.cuda.manual_seed_all(options.manual_seed)

cudnn.benchmark = True
def main():
    if options.wgan:
        wgan = WGAN(options)
        wgan.train()
    else:
        gan = GAN(options)
        gan.train()
if __name__ == '__main__':
    main()
