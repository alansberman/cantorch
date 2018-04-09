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

from torch.utils.data.sampler import SubsetRandomSampler


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

def get_length(data_dir):
    folders = os.listdir(data_dir)
    styles={}
    length = 0
    for f in folders:
        styles[f]=len(os.listdir(os.path.join(data_dir,f)))
    for k,value in styles.items():
        length += value
    return length

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




# Mille grazie https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb


def get_train_test_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
   
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = dset.ImageFolder(root=data_dir,transform=transform)
    dataset_length = get_length(data_dir)
    num_train = int((dataset_length)*(1-valid_size))
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return (train_loader, test_loader)


x,y = get_train_test_loader("D:\\WikiArt\\wikiart\\",4,3)
print(x)
print(y)
