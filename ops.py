import matplotlib.pyplot as plt
import random
import sys
import os
from PIL import Image
from glob import glob
from skimage import io, transform

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler


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
    count = 0
    for key, val in styles.items():
        styles[key] = count
        count += 1
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


def get_dataset(path):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = dset.ImageFolder(root=path,transform=transform)  
    return data

class WikiartDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Override

        Gets length of dataset

        """
        return len(os.listdir(self.root_dir))



    def __getitem__(self, idx ):
        """
        Override
            
        Gets the contents of a particular style

        """
        s = get_styles(self.root_dir)
        styles = {v:k for k,v in s.items()}
        style = styles[idx]
        
        #style = styles[index]
        images = os.listdir(os.path.join(self.root_dir,style))

       # sample = ({'style': )style, 'images': images}

        if self.transform:
            images = self.transform(images)

        return style,images

class StyleDataset(Dataset):
    def __init__(self, root_dir, style, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Annoying mapping from style string e.g. Realism to an int representing its position in the parent folder (e.g. 21)
        # Needed because pytorch *has* to have indexing in its datasets
        chosen_style = WikiartDataset(self.root_dir)
        s = get_styles(self.root_dir)
        idx = s[style]
        self.style, self.images_ = chosen_style[idx]


    def __len__(self):
        """
        Override

        Gets length of style 

        """
        return len(self.images_)
       


    def __getitem__(self, idx ):
        """
        Override
            
        Gets a particular image sample

        """
        path = os.path.join(self.root_dir,self.style)
        idx = str(idx)+".jpg"
        img_name = os.path.join(path,idx)
        image = io.imread(img_name)
        sample = {'image': image, 'style': self.style}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Thanks to https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def get_loss_graphs(train_history, path, model_name):

    x = range(len(train_history['gen_loss']))
    y_1 = train_history['disc_loss']
    y_2 = train_history['gen_loss']
    plt.plot(x,y_1,label="Discriminator Loss")
    plt.plot(x,y_2,label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path,model_name+"_loss_graph.png")
    plt.savefig(path)
    plt.close()


# Thanks to https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def get_class_loss_graph(train_history, path, model_name):

    x = range(len(train_history['gen_class_loss']))
    y_1 = train_history['disc_class_loss']
    y_2 = train_history['gen_class_loss']
    plt.plot(x,y_1,label="Discriminator Class Loss")
    plt.plot(x,y_2,label="Generator Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path,model_name+"_class_loss_graph.png")
    plt.savefig(path)
    plt.close()