import torch
import torchvision
import numpy as np
from image_folders import ImageFolder #, Dataset

class Witch:

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        self.args, self.setup = args, setup
        self.loss_fun = torch.nn.CrossEntropyLoss()
        
    def initialize_delta(self, ingredient):
        self.std_tensor = torch.tensor(ingredient.data_std)[None, :, None, None]
        self.mean_tensor = torch.tensor(ingredient.data_mean)[None, :, None, None]

        poison_delta = torch.randn(len(ingredient.poisonset), *ingredient.trainset[0][1].shape)
        poison_delta *= self.args.eps / self.std_tensor / 255
        poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps / (self.std_tensor * 255)),
                                             -self.args.eps / (self.std_tensor * 255))
        return poison_delta

    def calculate_loss(self, inputs, labels, victim, target_grad, target_grad_norm):
        norm_type = 2
        target_losses = 0 
        poison_norm = 0

        outputs = victim.model(inputs)
        poison_prediction = torch.argmax(outputs.data, dim=1)

        poison_correct = (poison_prediction == labels).sum().item()

        poison_loss = self.loss_fun(outputs, labels)
        poison_grad = torch.autograd.grad(poison_loss, victim.model.parameters(), retain_graph=True, create_graph=True)

        indices = torch.arange(len(poison_grad))

        for i in indices:
            target_losses -= (poison_grad[i] * target_grad[i]).sum()
            poison_norm += poison_grad[i].pow(2).sum()

        poison_norm = poison_norm.sqrt()

        # poison_grad_norm = torch.norm(torch.stack([torch.norm(grad, norm_type).to(device) for grad in poison_grad]), norm_type)
        target_losses /= target_grad_norm 
        target_losses = 1 + target_losses / poison_norm
        target_losses.backward()

        return target_losses.detach().cpu(), poison_correct

    def brew_poison(self, victim, ingredient, targets, intended_classes):

        poison_deltas = []
        adv_losses = []

        target_grad, target_grad_norm = victim.gradient(targets, intended_classes, self.loss_fun)
        
        if len(ingredient.poisonset) > 0:
            init_lr = 0.1
            for trial in range(self.args.restarts):            
                poison_delta = self.initialize_delta(ingredient)
                att_optimizer = torch.optim.Adam([poison_delta], lr=init_lr)

                poison_delta.grad = torch.zeros_like(poison_delta)
                poison_delta.requires_grad_()
                poison_bounds = torch.zeros_like(poison_delta)

                for iter in range(self.args.attackiter):
                    target_loss = 0
                    poison_correct = 0
                    for batch, example in enumerate(ingredient.poisonloader):
                        ids, inputs, labels = example

                        inputs = inputs.to(**self.setup)
                        labels = labels.to(**self.setup).long()
        
                        ### Finding the 
                        poison_slices, batch_positions = [], []
                        for batch_id, image_id in enumerate(ids.tolist()):
                            lookup = ingredient.poison_dict.get(image_id)
                            if lookup is not None:
                                poison_slices.append(lookup)
                                batch_positions.append(batch_id)
                                        
                        if len(batch_positions) > 0:
                            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
                            delta_slice.requires_grad_()
                            poison_images = inputs[batch_positions]
                            inputs[batch_positions] += delta_slice

                        loss, p_correct = self.calculate_loss(inputs, labels, victim, target_grad, target_grad_norm)
                        
                        # Update Step:
                        poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                        poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))

                        target_loss += loss
                        poison_correct += p_correct

                    att_optimizer.step()
                    att_optimizer.zero_grad()
  
                    with torch.no_grad():
                    # Projection Step 
                        poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps / self.std_tensor / 255), -self.args.eps / self.std_tensor / 255)
                        poison_delta.data = torch.max(torch.min(poison_delta, (1 - self.mean_tensor) / self.std_tensor - poison_bounds), -self.mean_tensor / self.std_tensor - poison_bounds)

                    if iter == self.args.attackiter - 1:
                        adv_losses.append(target_loss/(batch + 1))
                        poison_deltas.append(poison_delta)

            minimum_loss_trial = np.argmin(adv_losses)
            print("Trial #{} selected with target loss {}".format(minimum_loss_trial, adv_losses[minimum_loss_trial]))    
            return poison_deltas[minimum_loss_trial]

    def brew_camou(self, victim, ingredient, targets, true_classes):
        camou_deltas = []
        adv_losses = []
        std_tensor = torch.tensor(ingredient.data_std)[None, :, None, None]
        mean_tensor = torch.tensor(ingredient.data_mean)[None, :, None, None]

        
        target_grad, target_grad_norm = victim.gradient(targets, true_classes, self.loss_fun)
        
        if len(ingredient.camouset) > 0:
            init_lr = 0.1
            for trial in range(self.args.restarts):            
                camou_delta = self.initialize_delta(ingredient)
                att_optimizer = torch.optim.Adam([camou_delta], lr=init_lr)


                camou_delta.grad = torch.zeros_like(camou_delta)
                camou_delta.requires_grad_()
                poison_bounds = torch.zeros_like(camou_delta)

                for iter in range(self.args.attackiter):
                    target_loss = 0
                    camou_correct = 0
                    for batch, example in enumerate(ingredient.camouloader):
                        ids, inputs, labels = example

                        inputs = inputs.to(**self.setup)
                        labels = labels.to(**self.setup).long()
        
                        ### Finding the 
                        camou_slices, batch_positions = [], []
                        for batch_id, image_id in enumerate(ids.tolist()):
                            lookup = ingredient.camou_dict.get(image_id)
                            if lookup is not None:
                                camou_slices.append(lookup)
                                batch_positions.append(batch_id)
                                        
                        if len(batch_positions) > 0:
                            delta_slice = camou_delta[camou_slices].detach().to(**self.setup)
                            delta_slice.requires_grad_()
                            poison_images = inputs[batch_positions]
                            inputs[batch_positions] += delta_slice

                        loss, c_correct = self.calculate_loss(inputs, labels, victim, target_grad, target_grad_norm)
                        
                        # Update Step:
                        camou_delta.grad[camou_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                        poison_bounds[camou_slices] = poison_images.detach().to(device=torch.device('cpu'))

                        target_loss += loss
                        camou_correct += c_correct

                    att_optimizer.step()
                    att_optimizer.zero_grad()
  
                    with torch.no_grad():
                    # Projection Step 
                        camou_delta.data = torch.max(torch.min(camou_delta, self.args.eps / self.std_tensor / 255), -self.args.eps / self.std_tensor / 255)
                        camou_delta.data = torch.max(torch.min(camou_delta, (1 - mean_tensor) / self.std_tensor - poison_bounds), -mean_tensor / self.std_tensor - poison_bounds)

                    if iter == self.args.attackiter - 1:
                        adv_losses.append(target_loss/(batch + 1))
                        camou_deltas.append(camou_delta)

            minimum_loss_trial = np.argmin(adv_losses)
            print("Trial #{} selected with target loss {}".format(minimum_loss_trial, adv_losses[minimum_loss_trial]))    
            return camou_deltas[minimum_loss_trial]


    def brew(self, victim, ingredient, brewing_poison=True):

        targets = torch.stack([data[1] for data in ingredient.targetset], dim=0).to(**self.setup)
        intended_classes = torch.tensor([ingredient.poison_class]).to(**self.setup)
        true_classes = torch.tensor([data[2] for data in ingredient.targetset]).to(**self.setup)

        if brewing_poison:
            return self.brew_poison(victim, ingredient, targets, intended_classes)
        else:
            return self.brew_camou(victim, ingredient, targets, true_classes)

