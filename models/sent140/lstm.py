import torch
import torch.nn as nn
from utils.optim import get_optimizer, get_lr_scheduler
from ..model import Model


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        self.lstm.flatten_parameters()
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        _, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)


class LSTMSentiment(Model):
    def __init__(self, iterator, criterion, metric, device, optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3,
                 epoch_size=1, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True,
                 dropout=0.5):
        """

        :param iterator:
        :param criterion:
        :param metric:
        :param device:
        :param optimizer_name:
        :param lr_scheduler:
        :param initial_lr:
        :param embedding_dim:
        :param hidden_dim:
        :param output_dim:
        :param n_layers:
        :param bidirectional:
        :param dropout:
        """
        super(Model, self).__init__()

        self.device = device
        self.criterion = criterion
        self.metric = metric

        text_field = iterator.dataset.fields['text']

        pad_idx = text_field.vocab.stoi[text_field.pad_token]
        unk_idx = text_field.vocab.stoi[text_field.unk_token]

        self.net = LSTM(vocab_size=len(text_field.vocab),
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        n_layers=n_layers,
                        bidirectional=bidirectional,
                        dropout=dropout,
                        pad_idx=pad_idx).to(device)

        # initialize embeddings
        pretrained_embeddings = text_field.vocab.vectors
        self.net.embedding.weight.data.copy_(pretrained_embeddings)

        self.net.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim).to(self.device)
        self.net.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim).to(self.device)

        # Freeze embedding
        self.net.embedding.weight.requires_grad = False

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for batch in iterator:
            self.optimizer.zero_grad()

            text, text_lengths = batch.text

            predictions = self.net(text, text_lengths).squeeze(1)

            loss = self.criterion(predictions, batch.label)

            acc = self.metric(predictions, batch.label)

            loss.backward()

            self.optimizer.step()

            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        batch = next(iter(iterator))
        self.optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = self.net(text, text_lengths).squeeze(1)

        loss = self.criterion(predictions, batch.label)

        acc = self.metric(predictions, batch.label)

        loss.backward()

        if update:
            self.optimizer.step()
            self.lr_scheduler.step()

        return loss.item(), acc.item()

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()

        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text

                predictions = self.net(text, text_lengths).squeeze(1)

                loss = self.criterion(predictions, batch.label)

                acc = self.metric(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
