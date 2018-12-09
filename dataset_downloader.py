import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

image_size = 64
dataset = dset.CIFAR10(root= "D:\\WikiArt\\", download=True,
                           transform=transforms.Compose([
                               transforms.Scale(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )

