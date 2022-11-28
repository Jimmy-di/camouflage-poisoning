
import os
import torch
import torchvision
from image_folders import ImageFolder #, Dataset
from cifar_10 import CIFAR10

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np
import random
import math 

from Constants import CIFAR_mean, CIFAR_std, Imagenet_mean, Imagenet_std

class Ingredient:

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)) -> None:

        self.args, self.setup = args, setup
        traindir = None
        valdir = None
        self.num_classes = 10

        if args.dataset == 'Imagenette':
            traindir = os.path.join(args.datapath, 'train')
            valdir = os.path.join(args.datapath, 'val')
        elif args.dataset == 'Imagewoof':
            traindir = os.path.join(args.datapath, 'train')
            valdir = os.path.join(args.datapath, 'val')
        else:
            self.data_mean = (0.485, 0.456, 0.406)
            self.data_std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean=self.data_mean,std=self.data_std)

        if traindir:
            self.trainset = ImageFolder(
            traindir,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize,
            ]))

            self.validationset = ImageFolder(
            valdir,
                transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                normalize,
            ]))
        else:
            self.trainset = CIFAR10(train=True, transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ]))
            self.validationset = CIFAR10(train = False, transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            normalize,
                                                            ]))
        self.trainloader = DataLoader(self.trainset, batch_size=args.batchsize, shuffle=True)
        self.testloader = DataLoader(self.validationset, batch_size=args.batchsize, shuffle=False)
        
        

    def set_global_seed(self, seed):
        print("Setting seed as {}".format(seed))
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 3)
        np.random.seed(seed + 4)
        torch.cuda.manual_seed_all(seed + 5)
        random.seed(seed + 6)
        return
    
    def initialize_attack_setup(self):
        if self.args.seed:
            self.set_global_seed(self.args.seed)
        else:
            seed = np.random.randint(10000111)
            self.set_global_seed(seed)

        avail_classes = np.arange(self.num_classes)
        [self.target_class, self.poison_class] = np.random.choice(avail_classes, replace=False, size=2)
        camou_class = self.target_class

        # Choose Target

        target_indices = self.validationset.get_index(self.target_class)
        target_index = []
        target_index.append(np.random.choice(target_indices))

        self.targetset = Subset(self.validationset, target_index)
        self.targetloader = torch.utils.data.DataLoader(self.targetset)

        print("Target image is chosen with ID {}".format(target_index))

        # Choose Poison Images:

        poison_index = []

        poison_index = self.trainset.get_index(self.poison_class)
        number_poisons = math.floor(self.args.pbudget * len(self.trainset))

        if number_poisons > len(poison_index):
            number_poisons = len(poison_index)
            print("Poison budget is over the maximum limit and set to {}".format(number_poisons))

        poison_index = np.random.choice(poison_index, number_poisons, replace=False)
        self.poison_dict = {}

        for index, val in enumerate(poison_index):
            self.poison_dict[val] = index

        self.poisonset = Subset(self.trainset, poison_index)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=20, drop_last=False)


        # Choose Camouflage Images:
        camou_index = self.trainset.get_index(camou_class)
        number_camous = math.floor(self.args.cbudget * len(self.trainset))
        
        if number_camous > len(camou_index):
            number_camous = len(camou_index)
            print("Poison budget is over the maximum limit and set to {}".format(number_camous))

        camou_index = np.random.choice(camou_index, number_camous, replace=False)

        self.camou_dict = {}

        for index, val in enumerate(camou_index):
            self.camou_dict[val] = index

        self.camouset = Subset(self.trainset, camou_index)
        self.camouloader = torch.utils.data.DataLoader(self.camouset, batch_size=20, drop_last=False)
    