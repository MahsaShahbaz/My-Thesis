import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from opacus import PrivacyEngine
import logging


from seed_manager import set_seed
set_seed()


# Configure logging at the beginning of script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), label.clone().detach()


class LocalUpdate(object):
    def __init__(self, args, dataset, u_id, idxs):
        self.u_id = u_id
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.dataset_size = len(idxs)
        self.sample_rate = self.args.local_bs / self.dataset_size

    def train_val_test(self, dataset, idxs):
        _split = 0
        if self.args.local_test_split > 0.0:
            _split = max(int(np.round((self.args.local_test_split) * len(idxs))), 1)

        idxs_train = idxs[_split:]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        testloader = None
        if _split > 0:
            idxs_test = idxs[:_split]
            testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round, u_step=0):
        model.train()  # Ensure the model is in training mode.
        epoch_loss = []
        output_gradients = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        if self.args.withDP:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=self.args.virtual_batch_size,
                sample_size=self.dataset_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
            )
            privacy_engine.attach(optimizer)
            privacy_engine.steps = u_step

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
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args.delta)
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), optimizer.privacy_engine.steps, epsilon, output_gradients
        else:
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), 0, 0, output_gradients
