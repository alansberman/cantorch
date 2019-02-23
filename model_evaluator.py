# model_evaluator.py
# 26/11/2018

import argparse
import os
from generators import *
from sampler import *
from discriminators import *
from ops import *
from model import *
from wasserstein_model import *
from utils import *
import random
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import parfit.parfit as pf

#from tpot import TPOTClassifier
#import autosklearn.classification as ac
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import PIL
from hpsklearn import HyperoptEstimator, any_classifier
from hpsklearn import components as hpc
from hyperopt import tpe
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import warnings


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  help='cifar10 | lsun | imagenet | wikiart',default='cifar10')
parser.add_argument('--dataroot', help='path to dataset',default="D:\WikiArt\\") # D:\\WikiArt\\wikiart\\ #/mydata
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--gan_type', type=str, help='dcgan | wgan | can', default='can')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--optimizer', type=str, default="Adam", help='Adam | SGD | RMSProp')
parser.add_argument('--z_noise', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--y_dim',type=int,help="number of output/target classes",default=10)
parser.add_argument('--channels',type=int,help="number of image channels",default=3)
parser.add_argument('--num_gen_filters', type=int, default=64)
parser.add_argument('--power', type=int, default=8, help="1-number of hidden layers")
parser.add_argument('--disc_iterations', type=int, default=5,help="ratio of discriminator updates to generator")
parser.add_argument('--num_disc_filters', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5') 
parser.add_argument('--lower_clamp', type=float, default=-0.01, help='lower clamp for params')
parser.add_argument('--upper_clamp', type=float, default=0.01, help='upper clamp for params')
parser.add_argument('--cuda',  action='store_true',default=True, help='enables cuda')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gen_path', default='', help="path to netG (to continue training)")
parser.add_argument('--disc_path', default="C:\\Users\\alan\\Desktop\\experiments\\wdcgan_imagenet_64\\netD_epoch_24.pth", help="path to netD (to continue training)")
parser.add_argument('--out_folder', default="/output", help='folder to output images and model checkpoints') #"/output"
parser.add_argument('--wgan', type=bool, default=False, help='if training a la WGAN') #"/output"
parser.add_argument('--lsgan', type=bool, default=False, help='if training with LSGAN') #"/output"
parser.add_argument('--num_critic', type=int, default=5,help="D:G training") #"/output"
parser.add_argument('--gradient_penalty', type=bool, default=False,help="if training with WGANGP") #"/output"
parser.add_argument('--optimise', type=str, default="no_optimisation")
parser.add_argument('--svhn', type=bool, default=False, help='if training with LSGAN') #"/output"




# custom weights initialization called on self.generator and self.discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



    # thanks to https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
def weight_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def main():
    # Thanks to https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    directory = "C:\\Users\\alan\\Desktop\\experiments\\"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    options = parser.parse_args()
    # files = get_data(directory)
    # models = []
    # for f in files:
    #     if f.endswith(".pth"):
    #         models.append(f)
    ngpu = options.num_gpu
    channels = options.channels
    num_disc_filters = options.num_disc_filters

    if options.gan_type=="dcgan":
        if options.gradient_penalty and options.wgan:
            netD = GoodDiscriminator(32).to(device)
            print(netD)
        elif options.wgan:
            netD = WganDiscriminator(3,64).to(device)
            print(netD)
        else:
            netD = DiscriminatorOrig(1,3,64).to(device)

    netD.eval()

    if options.gradient_penalty:
        netD.apply(weight_init)
    else:
        netD.apply(weights_init)
    if options.disc_path != '':
        netD.load_state_dict(torch.load(options.disc_path))

    if options.gradient_penalty:
        layers = nn.Sequential(*list(netD.children()))
    else:
        layers = nn.Sequential(*list(netD.children()))[0]
    conv_layers = []
    # print(type(layers))
    # print(layers)
    # sys.exit(0)

    if options.gradient_penalty:
        for layer in layers:
            if type(layer)==MyConvo2d:
                conv_layers.append(*layer.children())
            if type(layer)==ResidualBlock:
                for item in list(layer.children()):
                    if type(item)==MyConvo2d:
                        conv_layers.append(*item.children())
    else:
        for layer in layers:
            if type(layer)== torch.nn.modules.conv.Conv2d:
                conv_layers.append(layer)
        # conv_layers.append(layers[0])
        # for layer in range(1,len(layers)+1):
        #     if type(layers[layer])==ResidualBlock:

        #         if type(item) == torch.nn.modules.conv.Conv2d:
        #             conv_layers.append(item)
    # print(netD)


    def get_conv_activations_wgangp(name,count):
        pool = nn.AdaptiveMaxPool2d(4)
        def hook(module, input, output):
            activations[str(name)+str(count)] = pool(output).view(output.size(0), -1)
        return hook
    def get_conv_activations(name):
        pool = nn.AdaptiveMaxPool2d(4)
        def hook(module, input, output):
            activations[name] = pool(output).view(output.size(0), -1)
        return hook

    # self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
    # self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down', hw=DIM)
    # self.rb2 = ResidualBlock(2*self.dim, 4*self.dim, 3, resample = 'down', hw=int(DIM/2))
    # self.rb3 = ResidualBlock(4*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/4))
    # self.rb4 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down', hw=int(DIM/8))
    # self.ln1 = nn.Linear(4*4*8*self.dim, 1)


    # Register forward hooks to get activations
    if options.gradient_penalty:
        netD.conv1.register_forward_hook(get_conv_activations_wgangp('conv',1))
        netD.rb1.register_forward_hook(get_conv_activations_wgangp('conv',2))
        netD.rb2.register_forward_hook(get_conv_activations_wgangp('conv',3))
        netD.rb3.register_forward_hook(get_conv_activations_wgangp('conv',4))
        netD.rb4.register_forward_hook(get_conv_activations_wgangp('conv',5))

    else:
        netD.main[0].register_forward_hook(get_conv_activations('conv1'))
        netD.main[2].register_forward_hook(get_conv_activations('conv2'))
        netD.main[5].register_forward_hook(get_conv_activations('conv3'))
        netD.main[8].register_forward_hook(get_conv_activations('conv4'))
        netD.main[11].register_forward_hook(get_conv_activations('conv5'))

    if options.dataset=='cifar10':
        training_data = dset.CIFAR10(root= options.dataroot, download=False,train=True,
                                transform=transforms.Compose([
                                    transforms.Resize(options.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
                    )
        test_data = dset.CIFAR10(root= options.dataroot, download=False,train=False,
                                transform=transforms.Compose([
                                    transforms.Resize(options.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
                    )
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1,
                                                shuffle=True, num_workers=int(options.workers))

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                            shuffle=True,
                                                num_workers=int(options.workers))

    else:
        # thanks to @Jordi_de_la_Torre https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
        svhn_train = dset.SVHN(root="D:\svhn",split="train",download=False, transform=transforms.Compose([
                                    transforms.Resize(options.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        svhn_test = dset.SVHN(root="D:\svhn",split="test",download=False, transform=transforms.Compose([
                                    transforms.Resize(options.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        train_dataloader = torch.utils.data.DataLoader(
            svhn_train, 
            shuffle=True,
            batch_size=1, 
            num_workers=int(options.workers), pin_memory=True) 

        test_dataloader = torch.utils.data.DataLoader(
            svhn_test, 
            sampler=ImbalancedDatasetSampler(svhn_test),
            batch_size=1, 
            num_workers=int(options.workers), pin_memory=True) 
    training_iter = iter(train_dataloader)
    test_iter = iter(test_dataloader)
    # print(len(train_dataloader),len(test_dataloader))
    i = 0
    x_train = []
    # Heavily inspired by https://github.com/pytorch/examples/blob/master/dcgan/main.py
    # x=[]
    y_train=[]
    while i < 50000:
        data = training_iter.next()
        data[0] = data[0].to(device)
        if options.gradient_penalty:
            data[0] = F.interpolate(data[0],64)
        activations = {}
        output = netD(data[0])

        get_conv_activations('conv1')
        get_conv_activations('conv2')
        get_conv_activations('conv3')
        get_conv_activations('conv4')
        get_conv_activations('conv5')
        act_flat = []
        y_train.append(data[1])
    
        for key in activations:
            act_flat.append(activations[key])
        act_flat = torch.cat(act_flat, 1)[0]
        x_train.append(act_flat.detach().cpu().numpy())
        i+=1

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    j=0
    x_test=[]
    y_test=[]

    if options.dataset=='svhn':
        test_length = len(test_dataloader)
    else:
        test_length = 10000
    while j < test_length:
        data = test_iter.next()
        data[0] = data[0].to(device)
        if options.gradient_penalty:
            data[0] = F.interpolate(data[0],64)
        activations = {}
        output = netD(data[0])
        get_conv_activations('conv1')
        get_conv_activations('conv2')
        get_conv_activations('conv3')
        get_conv_activations('conv4')
        get_conv_activations('conv5')
        act_flat = []
        y_test.append(data[1])
        for key in activations:
            act_flat.append(activations[key])
        act_flat = torch.cat(act_flat, 1)[0]
        x_test.append(act_flat.detach().cpu().numpy())
        j+=1

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)   
    title = str(options.gan_type+"_wgan-"+str(options.wgan)+"_"+options.dataset+"_"+options.optimise+"_gradpen-"+str(options.gradient_penalty)+".txt")
    output_file = open(title,"w")
    if options.optimise != 'no_optimisation':
        model = HyperoptEstimator(classifier=hpc.sgd('_sgd',penalty='l2',n_jobs=-1),max_evals=25,trial_timeout=1200)
    else:
        model = linear_model.SGDClassifier(penalty='l2',n_jobs=-1)
    # model = HyperoptEstimator(classifier=any_classifier('my_clf'),
    #                       algo=tpe.suggest,
    #                       max_evals=10,
    #                       trial_timeout=30)
    #model = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    # model = ac.AutoSklearnClassifier()
    #model = linear_model.
    #model = linear_model.SGDClassifier(n_jobs=-1)
    # grid = {
    #     'alpha': [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3],
    #     'max_iter': [1000],
    #     'loss': ['hinge'],
    #     'penalty': ['l2'],
    #     'n_jobs': [-1]
    # }
    # param_grid = ParameterGrid(grid)
    # best_model, best_score, all_models, all_scores = pf.bestFit(linear_model.SGDClassifier,param_grid,x_train,y_train,
    # x_test,y_test,metric=accuracy_score,scoreLabel='Acc')
    # print(best_model,best_score)
    model.fit(x_train,y_train)
    # #print(model,"is the model")
    # predictions = model.predict(x_test)
    # print(predictions,"are the predictions")
    score = model.score(x_test,y_test)
    output_file.write("Score:\n"+str(score))
    if options.optimise != 'no_optimisation':
        best = str(model.best_model())
        output_file.write("\nBest model:\n"+str(best))
    else:
        params = str(model.get_params())
        output_file.write("\nModel parameters:\n"+str(params))
    # score = model.score(x_test,y_test)
    
    #output_file.write("\n"+str(best))
    # output_file.write(str(np.asarray(predictions))+"\n")
    # output_file.write(str(np.asarray(y_test))+"\n")
    output_file.close()
    #model.export('tpot_mnist_pipeline.py')


if __name__=="__main__":
    main()