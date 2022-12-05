# Run a single experiment

from tools.arg_parse import options
from tools.models import get_model
from forest.ingredients import Ingredient
from forest.victim import Victim
from forest.witch import Witch

import torch

import datetime
import time
from torchvision.utils import save_image
import sys

from copy import deepcopy


# Parse input arguments

args = options().parse_args()


if __name__ == "__main__":
    start_time = time.time()

    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)

    ingredients = Ingredient(args, setup=setup)
    ingredients.initialize_attack_setup()

    # Train model on clean images

    victim = Victim(args, setup=setup)
    victim.initialize_victim()
    victim.train(ingredients)

    # Obtain Poison
    witch = Witch(args, setup=setup)
    poison_delta = witch.brew(victim, ingredients, True)

    # Train model on clean + poisoned images to evaluate poisons
    victim.retrain(ingredients)
    
    # Obtain Camouflages

    camou_delta = witch.brew(victim, ingredients, False)

    # Train model on clean + poisoned + camou images to evaluate camouflages
    victim.retrain(ingredients)

    #save(ingredients, poison_delta, camou_delta)
    print("Ends here") 
    print("--- %s seconds ---" % (time.time() - start_time))