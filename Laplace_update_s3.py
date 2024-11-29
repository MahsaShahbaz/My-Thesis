import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.distributions import Laplace
from torch.nn.utils import clip_grad_norm_
from seed_manager import set_seed
set_seed()

def add_laplace_noise(tensor, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = torch.distributions.Laplace(0, scale).sample(tensor.shape).to(tensor.device)
    return tensor + noise

class LaplaceOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults, sensitivity, epsilon, max_grad_norm):
        super().__init__(params, defaults)
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(param, self.max_grad_norm)
                
                # Add Laplace noise to the gradients
                param.grad = add_laplace_noise(param.grad, self.sensitivity, self.epsilon)
                param.data.add_(param.grad, alpha=-group['lr'])

        return loss


class DatasetSplit(Dataset):
    """
    Dataset class for splitting datasets based on given indices.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), label




class LocalUpdate(object):
    def __init__(self, args, dataset, u_id, idxs):
        self.args = args
        self.device = args.device
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset and user indexes.
        """
        _split = 0
        if self.args.local_test_split > 0.0:
            _split = max(int(np.round(self.args.local_test_split * len(idxs))), 1)

        idxs_train = idxs[_split:]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        testloader = None
        if _split > 0:
            idxs_test = idxs[:_split]
            testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                    batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round, u_step=0):
        model.train()
        epoch_loss = []
        output_gradients = []
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, 
                                        momentum=self.args.momentum)        
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        
        if self.args.withDP:
            # Custom Laplace Optimizer
            optimizer = LaplaceOptimizer(
                model.parameters(), 
                {'lr': self.args.lr}, 
                sensitivity=self.args.sensitivity,
                epsilon=self.args.epsilon,
                max_grad_norm=self.args.max_grad_norm  # Include max_grad_norm for gradient clipping
            )

        for iter in range(self.args.local_ep):
            batch_loss = []
            optimizer.zero_grad()
            if self.args.withDP:
                virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                model_preds = model(images)
                loss = self.criterion(model_preds, labels)
                loss.backward()

                # Capture gradients of the output layer
                if model.classifier[1].weight.grad is not None:
                    output_gradients.append(model.classifier[1].weight.grad.clone().detach().cpu())
                if model.classifier[1].bias.grad is not None:
                    output_gradients.append(model.classifier[1].bias.grad.clone().detach().cpu())

                if self.args.withDP:
                    # take a real optimizer step after N_VIRTUAL_STEP steps t                                        
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        optimizer.step()
                        optimizer.zero_grad()                        
                    else:                        
                        optimizer.virtual_step() # take a virtual step                        
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if self.args.withDP:
            epsilon = self.args.epsilon
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), output_gradients, epsilon
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), output_gradients
