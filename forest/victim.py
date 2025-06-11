from tools.models import get_model
import os 
import torch
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
class Victim:

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        self.initialize_victim()

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)
            
    def initialize_victim(self, clean_training=True):
        self.model = get_model(self.args.net).to(**self.setup)
        self.epochs = self.args.epochs
        
        self.optimizer = torch.optim.SGD(params = self.model.parameters(), lr = self.args.eta, weight_decay = 5e-4, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        self.loss_fun = torch.nn.CrossEntropyLoss().cuda()

        self.model_path = self.args.model_path

    def train(self, ingredients):

        intended_classes = torch.tensor([ingredients.poison_class]).to(**self.setup)
        true_classes = torch.tensor([data[2] for data in ingredients.targetset]).to(**self.setup)

        if self.args.load_model:
            os.makedirs(self.model_path, exist_ok=True)
            path = os.path.join(self.model_path, "{}_{}.pth".format(self.args.dataset, self.args.net))
            self.model.load_state_dict(torch.load(path))

            self.validate(0, ingredients)
            self.check_target(ingredients, true_classes, intended_classes)

        else:
              # Fit
            for epoch in range(self.epochs):
                
                train_loss = []

                correct_preds = 0
                total_preds = 0
                
                for index, inputs, labels in ingredients.trainloader:

                    inputs, labels = inputs.to(**self.setup), labels.to(**self.setup)

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
    
                if epoch % 2 == 0:
                    self.validate(epoch, ingredients)
                    self.check_target(ingredients, true_classes, intended_classes)
            
                self.scheduler.step()


        if self.args.save_model:
            os.makedirs(self.model_path, exist_ok=True)
            path = os.path.join(self.model_path, "{}_{}.pth".format(self.args.dataset, self.args.net))
            torch.save(self.model.state_dict(), path)

    def validate(self, epoch, ingredients):

        valid_losses = []
        correct = 0
        total = 0
        self.model.eval()

        for _, inputs, labels in ingredients.testloader:

            labels = labels.type(torch.LongTensor)
        # Validate on Testloader
            inputs, labels = inputs.to(**self.setup), labels.to(**self.setup)

            with torch.no_grad():
                output = self.model(inputs)
            
                valid_loss = self.loss_fun(output, labels)
                valid_losses.append(valid_loss.item())

                predictions = torch.argmax(output.data, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        
        print("Validation Epoch {}: Loss: {}, Accuracy: {}".format(epoch, np.mean(valid_losses), correct / total))
        return

    def check_target(self, ingredients, true_classes, intended_classes):

        for index, inputs, labels in ingredients.targetloader:
            inputs, labels = inputs.to(**self.setup), labels.to(**self.setup).long()
                    
            with torch.no_grad():
                output = self.model(inputs)
                print(output)
                predictions = torch.argmax(output.data, dim=1)
      
                if predictions[0] == true_classes[0]:
                    print("Target is not fooled.")
                elif predictions[0] == intended_classes[0]:
                    print("Target is fooled.")
                else:
                    print("Target classfied incorrectly.")

    def gradient(self, images, labels, loss_fun):

        target_grad = 0
        target_gnorm = 0
        
        labels = labels.long()

        loss = loss_fun(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
  
        for grad in gradients:
            target_gnorm += grad.detach().pow(2).sum()
        target_gnorm = target_gnorm.sqrt()
        return gradients, target_gnorm


    def retrain(self, ingredients,poison_delta=None, camou_delta=None):
        self.initialize_victim(clean_training=False)

        targets = torch.stack([data[1] for data in ingredients.targetset], dim=0).to(**self.setup)
        intended_classes = torch.tensor([ingredients.poison_class]).to(**self.setup)
        true_classes = torch.tensor([data[2] for data in ingredients.targetset]).to(**self.setup)

        for epoch in range(self.args.epochs):
  
            print("Begin Re-training epoch {}:".format(epoch))

            train_loss = []

            correct_preds = 0
            total_preds = 0
  
            for index, inputs, labels in ingredients.trainloader:
                self.model.train()

                inputs, labels = inputs.to(**self.setup), labels.to(**self.setup)
                self.optimizer.zero_grad()            # reset the gradients to zero

                picture_cid = []
                camou_order = []

                picture_id = []
                poison_order = []

            # Use poison_dict to match poison_delta[i] to the correct poison image:
                if poison_delta is not None:
                    for order, id in enumerate(index.tolist()):
                        #print(ingredients.poison_dict)
                        if ingredients.poison_dict.get(id) is not None:
                            picture_id.append(order)
                            poison_order.append(ingredients.poison_dict[id])

                    if len(poison_order) > 0:
                        inputs[picture_id] += poison_delta[poison_order].to(**self.setup)
                
                if camou_delta is not None:
                    for order, id in enumerate(index.tolist()):
                        if ingredients.camou_dict.get(id) is not None:
                            picture_cid.append(order)
                            camou_order.append(ingredients.camou_dict[id])

                    if len(camou_order) > 0:
                        inputs[picture_cid] += camou_delta[camou_order].to(**self.setup)

    
                output = self.model(inputs)            # Generate model outputs
                loss = self.loss_fun(output, labels)   # Calculate loss

                loss.backward()            # Compute gradients
                self.optimizer.step()            # update parameters,

                predictions = torch.argmax(output.data, dim=1)

                total_preds += labels.size(0)
                correct_preds += (predictions == labels).sum().item()

                train_loss.append(loss.item())

            print("Training Epoch {}: Loss: {}, Accuracy: {}".format(epoch, np.mean(train_loss), correct_preds / total_preds))
            
            # validation phase - once every 10 epochs
      
            if epoch % 10 == 0:
                self.model.eval()
                valid_losses = []
                correct = 0
                total = 0
    
                self.check_target(ingredients, true_classes, intended_classes)
                self.validate(epoch, ingredients)
                self.model.train()
            self.scheduler.step()

        return



