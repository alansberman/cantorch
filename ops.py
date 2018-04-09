import matplotlib.pyplot as plt
import random
import sys
import os
from PIL import Image
from glob import glob

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

import numpy as np



# functions to show an image\
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_classes(data_dir):
    folders = os.listdir(data_dir)
    styles={}
    for f in folders:
        styles[f]=len(os.listdir(os.path.join(data_dir,f)))
    return list(styles.keys())

def get_styles(data_dir):
    folders = os.listdir(data_dir)
    styles={}
    for f in folders:
        styles[f]=len(os.listdir(os.path.join(data_dir,f)))
    return styles        

def get_data(data_dir):
    data = glob(data_dir+"/*/*", recursive=True)
    return data

def get_labels(data):
    labels = []
    for i in range(len(data)):
        start_idx = data[i].find("art\\")+4
        modded = data[i][start_idx:]
        stop_idx = modded.find("\\")
        labels.append(modded[:stop_idx])
    return labels

def get_image(image_path, mode):
    image = Image.open(image_path)
    return np.array(image.convert(mode))
# ty gangogh

def get_train_and_test_sets(styles,split_percentage):
    train_set_image_names = {}
    train_set_image_names = {}
    for key,value in styles.items():
        number_per_style = range(value)
        random.shuffle(list(number_per_style))
        train_set_image_names[key] = number_per_style[:value//split_percentage]
        train_set_image_names[key] = number_per_style[value//split_percentage:]
    return train_set_image_names, train_set_image_names

def test():
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = dset.ImageFolder(root="D:\\WikiArt\\wikiart\\",transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True)


#    testset = dset.ImageFolder(root='tests',transform=transform)

    classes= get_classes("D:\\WikiArt\\wikiart\\")

    

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(vutils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

test()