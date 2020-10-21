import torch
import torch.nn as nn
from utils.optim import get_optimizer, get_lr_scheduler
from torchvision.models import resnet18
from ..model import Model

NUMBER_CLASSES = 80


class INaturalistCNN(Model):
    def __init__(self, criterion, metric, device,
                 optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1, coeff=1):
        super(Model, self).__init__()

        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, NUMBER_CLASSES)
        self.net = self.net.to(device)
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.coeff = coeff

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            predictions = self.net(x)

            loss = self.coeff * self.criterion(predictions, y)

            acc = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        self.optimizer.zero_grad()

        predictions = self.net(x)

        loss = self.criterion(predictions, y)

        acc = self.metric(predictions, y)

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()

        with torch.no_grad():
            for x, y in iterator:
                predictions = self.net(x)

                loss = self.criterion(predictions, y)

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)