import torch
from torchtext import data


def get_iterator_sent140(path, all_data_path, device, max_vocab_size=25_000, batch_size=64):
    """
    Build text iterator to be use with LSTM model,
    :param path: path to .json file used to build the iterator, see TorchText for .json file format.
    :param all_data_path: path to .json file containing all train data
    :param device:
    :param max_vocab_size:
    :param batch_size:
    :return: iterator over sent140 samples, each sample has two attributes "text" and "label"
    """
    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

    text_data = data.TabularDataset(path=path, format='json', fields=fields)

    text_data.sort_key = lambda x: len(x.text)

    # Fix the seed
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_text_data = data.TabularDataset(path=all_data_path, format='json', fields=fields)

    # vocab is built using all data, in order to have the same mapping from words to indexes across workers
    TEXT.build_vocab(all_text_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(text_data)

    iterator = data.BucketIterator(
        text_data,
        batch_size=batch_size,
        sort_within_batch=True,
        device=device)

    return iterator

