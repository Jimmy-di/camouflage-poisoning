from arg_parse import options
from models import get_model

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

    MODEL = get_model(args)
    print(MODEL)

    print(args)
    print("Ends here") 
    print("--- %s seconds ---" % (time.time() - start_time))