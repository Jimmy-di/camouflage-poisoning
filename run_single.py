# Run a single experiment

from arg_parse import options
from models import get_model
from ingredients import Ingredient
from victim import Victim
from kettle import Kettle

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

    device = torch.device('cpu')
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Running on a GPU")
    else:
        print("Running on CPU...Are you sure you want to do this?")

    ingredients = Ingredient(args)
    ingredients.initialize_attack_setup(args)

    victim = Victim()
    victim.initialize_victim(args)
    victim.train(ingredients, args.loadmodel, args.savemodel, device)

    kettle = Kettle()
    poison_delta = kettle.brew(victim, ingredients)

    print("Ends here") 
    print("--- %s seconds ---" % (time.time() - start_time))