import torch
from torch.utils.data import Dataset, DataLoader
import string


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len, device):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        :param device:
        """
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len
        self.device = device
        f = open(file_path, 'r')
        self.text = f.read()

    def __len__(self):
        return len(self.text) // (self.chunk_len + 1)

    def __getitem__(self, idx):
        input_ = torch.zeros(self.chunk_len).long()
        for c in range(self.chunk_len):
            input_[c] = self.all_characters.index(self.text[idx + c])

        target = torch.zeros(self.chunk_len).long()
        for c in range(self.chunk_len):
            target[c] = self.all_characters.index(self.text[idx + c + 1])

        return input_.to(self.device), target.to(self.device)


def get_iterator_shakespeare(file_path, device, batch_size, chunk_len=200):
    """
    get next character prediction DataLoader, yields `batch_size` batches of `CharacterDataset` samples
    :param file_path: path to .txt file containing the training corpus
    :param chunk_len: (int) the length of the input and target sequences
    :param device:
    :param batch_size
    :return: iterator over shakespeare dataset samples
    """
    dataset = CharacterDataset(file_path, chunk_len, device)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator
