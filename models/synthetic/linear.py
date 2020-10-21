import torch
import torch.nn as nn
from utils.optim import get_lr_scheduler, get_optimizer
from ..model import Model


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes)

    def forward(self, x):
        return self.fc(x)


class LinearModel(Model):
    def __init__(self, criterion, metric, device, input_dimension, num_classes,
                 optimizer_name="adam", lr_scheduler="cyclic", initial_lr=1e-3, epoch_size=1):
        super(Model, self).__init__()

        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.net = LinearLayer(input_dimension, num_classes).to(self.device)

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()

            predictions = self.net(x)

            loss = self.criterion(predictions, y.float())

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

        loss = self.criterion(predictions, y.float())

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

                loss = self.criterion(predictions, y.float())

                acc = self.metric(predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

