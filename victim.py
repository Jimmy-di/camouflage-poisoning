from models import get_model

class Victim():
    def __init__(self) -> None:
        pass

    def initialize_victim(self, args):
        self.model = get_model(args)

    def train(self, ingredients, loadmodel):
        return 0

    def validate(self, args, ingredients):
        return 0


