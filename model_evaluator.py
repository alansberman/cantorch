# # model_evaluator.py
# # 26/11/2018

# import argparse
# import os
# from generators import *
# from discriminators import *
# from ops import *
# from model import *
# from wasserstein_model import *
# from utils import *
# import random
# import torch

# import numpy as np
# import sys
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# cudnn.benchmark=True

# import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
# from torch.autograd import Variable
# import time


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',  help='cifar10 | lsun | imagenet | wikiart',default='cifar10')
# parser.add_argument('--dataroot', help='path to dataset',default="D:\\WikiArt\\") # D:\\WikiArt\\wikiart\\ #/mydata
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
# parser.add_argument('--gan_type', type=str, help='dcgan | wgan | can', default='can')
# parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
# parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
# parser.add_argument('--optimizer', type=str, default="Adam", help='Adam | SGD | RMSProp')
# parser.add_argument('--z_noise', type=int, default=25, help='size of the latent z vector')
# parser.add_argument('--y_dim',type=int,help="number of output/target classes",default=10)
# parser.add_argument('--channels',type=int,help="number of image channels",default=3)
# parser.add_argument('--num_gen_filters', type=int, default=64)
# parser.add_argument('--power', type=int, default=8, help="1-number of hidden layers")
# parser.add_argument('--disc_iterations', type=int, default=5,help="ratio of discriminator updates to generator")
# parser.add_argument('--num_disc_filters', type=int, default=64)
# parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5') 
# parser.add_argument('--lower_clamp', type=float, default=-0.01, help='lower clamp for params')
# parser.add_argument('--upper_clamp', type=float, default=0.01, help='upper clamp for params')
# parser.add_argument('--cuda',  action='store_true',default=True, help='enables cuda')
# parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--gen_path', default='', help="path to netG (to continue training)")
# parser.add_argument('--disc_path', default='', help="path to netD (to continue training)")
# parser.add_argument('--out_folder', default="/output", help='folder to output images and model checkpoints') #"/output"
# parser.add_argument('--wgan', type=bool, default=False, help='if training a la WGAN') #"/output"
# parser.add_argument('--lsgan', type=bool, default=False, help='if training with LSGAN') #"/output"
# parser.add_argument('--num_critic', type=int, default=5,help="D:G training") #"/output"
# parser.add_argument('--gradient_penalty', type=bool, default=False,help="if training with WGANGP") #"/output"


# parser.add_argument('--manual_seed', type=int, help='manual seed')
# directory = "C:\\Users\\alan\\Desktop\\experiments\\"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# options = parser.parse_args()
# files = get_data(directory)
# models = []
# for f in files:
#     if f.endswith(".pth"):
#         models.append(f)

# def main():
#     discriminator = Can64Discriminator(options.channels, options.y_dim, options.num_disc_filters).to(device)
#     criterion = nn.BCELoss()
#     style_criterion = nn.CrossEntropyLoss()
#     m = torch.load(models[0])
#     discriminator.load_state_dict(m)
#     labels = torch.full((options.batch_size,), 1, device=device)

#     # Generator class/style labels
#     gen_style_labels = torch.LongTensor(options.batch_size)
#     gen_style_labels = gen_style_labels.fill_(1)
#     #results = open("classification_accuracy.txt","w")
#     if options.dataset == 'cifar10':
#         data = dset.CIFAR10(root= options.dataroot, download=True,
#                         transform=transforms.Compose([
#                             transforms.Resize(options.image_size),
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                         ])
#         )
#     else:
#         data = get_dataset(options.dataroot)
       
#         # # Set the type of GAN
#     dataloader = torch.utils.data.DataLoader(data, batch_size=options.batch_size,shuffle=True, num_workers=int(options.workers))
#     # Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py
#     data_iterator = iter(dataloader)
#     i = 0
#     num_batches = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in dataloader:
#             real_images, image_labels = data
#             print(image_labels)
#             real_images = real_images.to(device) 
#             outputs, predictions = discriminator(real_images)
#             print(predictions)
#             total += labels.size(0)
#             correct += (predictions == image_labels).sum().item()
#             batch_size = real_images.size(0)
#             # loss = 0
#             # class_loss = 0

#             # real_image_labels = torch.LongTensor(batch_size).to(device)
#             # real_image_labels.copy_(image_labels)
#             # labels = torch.full((batch_size,), 1, device=device)


#             # predicte
#             # label smoothing
#             # rand_labels = np.random.uniform(low=0.7, high=1.2, size=(batch_size,))
#             # r_labels = torch.from_numpy(rand_labels)
#             # labels.copy_(r_labels)
#             #print(labels)
#             # if options.gan_type == 'can':
#             #     predicted_output_real, predicted_styles_real = discriminator(real_images.detach())
#             #     total+=batch_size
#             #     correct += (predicted_output_real==image_labels).sum().item()
#             #     disc_class_loss = style_criterion(predicted_styles_real,real_image_labels)
#             #     results.write("Batch "+str(i)+" style classification loss:"+str(disc_class_loss.mean().item())+"\n")

#             # else:
#             #     predicted_output_real = options.discriminator(real_images.detach())
            
#             # disc_loss_real = criterion(predicted_output_real,labels).mean().item()
#             # results.write("Batch "+str(i)+" real/fake loss:"+str(disc_loss_real)+"\n")
#             # i+=1
#     print(100* (correct/total))


# if __name__ == '__main__':
#     main()
