import torch


def binary_accuracy(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def accuracy(preds, y):
    """

    :param preds:
    :param y:
    :return:
    """
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc