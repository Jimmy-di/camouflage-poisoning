from models import get_model

import torch
import numpy as np

class Victim():

    def initialize_victim(self, args):
        self.model = get_model(args)
        self.epochs = args.epochs
        self.optimizer = torch.optim.SGD(params = self.model.parameters(), lr = args.eta, weight_decay = 5e-4, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.model_path = args.model_path

    def train(self, ingredients, loadmodel, savemodel, device):

        if not loadmodel:
              # Fit
            for epoch in range(self.epochs):
                train_loss = []

                correct_preds = 0
                total_preds = 0
                self.model.train()
                for index, inputs, labels in ingredients.trainloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    self.optimizer.zero_grad()            # reset the gradients to zero
                    outputs = self.model(inputs)            # Generate model outputs
                    loss = self.loss_fun(outputs, labels)   # Calculate loss
                    loss.backward()            # Compute gradients
                    self.optimizer.step()            # update parameters,

                    predictions = torch.argmax(outputs.data, dim=1)

                    total_preds += labels.size(0)
                    correct_preds += (predictions == labels).sum().item()

                    train_loss.append(loss.item())

                print("Training Epoch {}: Loss: {}, Accuracy: {}".format(epoch, np.mean(train_loss), correct_preds / total_preds))
    
                if epoch % 10 == 0:
                    self.validate(ingredients, device)
                    self.check_target(ingredients, device)
            
                self.scheduler.step()

        else:
            self.model.load_state_dict(torch.load(self.model_path))
            self.validate(ingredients, device)
            self.check_target(ingredients, device)
        if savemodel:
            torch.save(self.model.state_dict(), self.model_path)

    def validate(self, ingredients, device):

        valid_losses = []
        correct = 0
        total = 0
        self.model.eval()

        for index, inputs, labels in ingredients.validationloader:
        # Validate on Testloader
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                output = self.model(inputs)
            valid_loss = self.loss_fun(output, labels)
            valid_losses.append(valid_loss.item())
            predictions = torch.argmax(output.data, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    def check_target(self, ingredients, device):
        for index, inputs, labels in ingredients.targetloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                output = self.model(inputs)
                print(output)


