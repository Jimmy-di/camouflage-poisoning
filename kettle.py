import torch
import torchvision
from image_folders import ImageFolder #, Dataset

class Kettle():

    def __init__(self, poison = True) -> None:
        self.poison = poison

    def poison_setup(self, args):
        return 0

    def calculate_loss(self):
        loss = 0
        return loss

    def brew(self, victim, ingredient):
        return

