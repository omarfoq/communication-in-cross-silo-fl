import torch
import torch.nn as nn
from utils.optim import get_optimizer, get_lr_scheduler
from torch.autograd import Variable
import string
from ..model import Model


class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_, hidden):
        self.gru.flatten_parameters()
        batch_size = input_.size(0)
        encoded = self.encoder(input_)
        output, hidden = self.gru(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


class NextCharDecoder(Model):
    def __init__(self, criterion, metric, device,
                 optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1,
                 embed_size=16, hidden_size=256, n_layers=2):
        super(Model, self).__init__()

        vocab_size = len(string.printable)
        self.net = RNN(vocab_size, embed_size, hidden_size, vocab_size, n_layers).to(device)
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        self.net.train()

        epoch_loss = 0
        epoch_acc = 0

        for inp, target in iterator:

            inp = inp.to(self.device)
            target = target.to(self.device)

            hidden = self.net.init_hidden(inp.size(0)).to(self.device)
            self.optimizer.zero_grad()

            loss = 0
            acc = 0

            for c in range(iterator.dataset.chunk_len):
                output, hidden = self.net(inp[:, c], hidden)
                loss += self.criterion(output.view(inp.size(0), -1), target[:, c])
                acc += self.metric(output, target[:, c]).item()

            loss /= iterator.dataset.chunk_len
            acc /= iterator.dataset.chunk_len

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        inp, target = next(iter(iterator))
        inp = inp.to(self.device)
        target = target.to(self.device)

        hidden = self.net.init_hidden(inp.size(0)).to(self.device)
        self.optimizer.zero_grad()

        loss = 0
        acc = 0

        for c in range(iterator.dataset.chunk_len):
            output, hidden = self.net(inp[:, c], hidden)
            loss += self.criterion(output.view(inp.size(0), -1), target[:, c])
            acc += self.metric(output, target[:, c]).item()

        loss /= iterator.dataset.chunk_len
        acc /= iterator.dataset.chunk_len

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        return loss.item(), acc

    def evaluate_iterator(self, iterator):
        self.net.eval()

        epoch_loss = 0
        epoch_acc = 0

        for inp, target in iterator:

            inp = inp.to(self.device)
            target = target.to(self.device)

            hidden = self.net.init_hidden(inp.size(0)).to(self.device)

            loss = 0
            acc = 0
            for c in range(iterator.dataset.chunk_len):
                output, hidden = self.net(inp[:, c], hidden)
                loss += self.criterion(output.view(inp.size(0), -1), target[:, c])
                acc += self.metric(output, target[:, c]).item()

            loss /= iterator.dataset.chunk_len
            acc /= iterator.dataset.chunk_len

            epoch_loss += loss.item()
            epoch_acc += acc
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def generate(self, prime_str="Wh", predict_len=200, temperature=0.8):
        all_characters = string.printable
        hidden = self.net.init_hidden(1).to(self.device)

        prime_input = torch.zeros(1, len(prime_str)).long().to(self.device)
        for c in range(len(prime_str)):
            prime_input[0, c] = all_characters.index(prime_str[c])

        predicted = prime_str

        for p in range(len(prime_str) - 1):
            _, hidden = self.net(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self.net(inp, hidden)

            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            predicted_char = all_characters[top_i]
            predicted += predicted_char

            inp = torch.zeros(1, len(predicted_char)).long().to(self.device)
            for c in range(len(predicted_char)):
                inp[0, c] = all_characters.index(predicted_char[c])

        return predicted
