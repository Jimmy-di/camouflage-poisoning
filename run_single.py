# Run a single experiment

from tools.models import get_model
from forest.ingredients import Ingredient
from forest.victim import Victim
from forest.witch import Witch

import torch

import datetime
import time
from torchvision.utils import save_image
import sys
import argparse
from copy import deepcopy


def main(args):
    start_time = time.time()

    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    setup = dict(device=device)

    ingredients = Ingredient(args, setup=setup)
    ingredients.initialize_attack_setup()

    # Train model on clean images

    victim = Victim(args, setup=setup)
    victim.initialize_victim()
    victim.train(ingredients)

    print("Clean Training Done")
    clean_train_time = time.time()
    print("--- %s seconds ---" % (clean_train_time - start_time))

    # Obtain Poison
    witch = Witch(args, setup=setup)
    poison_delta = witch.brew(victim, ingredients, True)

    print("Poisoning Done")
    poisoning_time = time.time()
    print("--- %s seconds ---" % (poisoning_time - clean_train_time))
    # Train model on clean + poisoned images to evaluate poisons
    victim.retrain(ingredients, poison_delta)
    
    # Obtain Camouflages
    pre_camou_time = time.time()
    camou_delta = witch.brew(victim, ingredients, False)

    print("Camouflage Done")
    camou_time = time.time()
    print("--- %s seconds ---" % (camou_time - pre_camou_time))

    # Train model on clean + poisoned + camou images to evaluate camouflages
    victim.retrain(ingredients, poison_delta, camou_delta)

    #save(ingredients, poison_delta, camou_delta)
    print("Total Time:") 
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    # Add an argument

    # Training / General Parameter
    parser.add_argument('--seed', type=int, default=100001111)
    parser.add_argument('--net', type=str, default="ResNet18")
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eta', default=0.05, type=float)  # randn / rand

    parser.add_argument('--no_augment', action='store_true', help='Use data augmentation to increase robustness')
    # Save and Load Models
    parser.add_argument('--save_model', action='store_true', help='Save trained model to model_path')
    parser.add_argument('--load_model', action='store_true', help='Load trained model from model_path')
    parser.add_argument('--model_path',  default='model_checkpoint/', type=str)
    
    # Poisoning Param:
    parser.add_argument('--attackoptim', default='Adam', type=str)
    parser.add_argument('--attackiter', default=251, type=int)
    
    parser.add_argument('--poisontype', default='witches', type=str)  # randn / rand
    # Poison brewing:

    parser.add_argument('--pbudget', type=float, default=0.01)
    parser.add_argument('--pinit', default='randn', type=str)  # randn / rand
    parser.add_argument('--prestarts', default=8, type=int, help='How often to restart the attack.')
    parser.add_argument('--ptargets', default=1, type=int, help='Number of targets')
    parser.add_argument('--pepsilon', type=int, default=16)

    # Camouflage brewing:
    parser.add_argument('--cbudget', type=float, default=0.01)
    parser.add_argument('--cinit', default='randn', type=str)  # randn / rand
    parser.add_argument('--crestarts', default=8, type=int, help='How often to restart the attack.')
    parser.add_argument('--ctargets', default=1, type=int, help='Number of targets')
    parser.add_argument('--cepsilon', type=int, default=16)

    # Parse the argument
    args = parser.parse_args()    
    main(args)

 