import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from PIL import Image


class INaturalist(Dataset):
    def __init__(self, pickle_file, root_path, device, transforms=None):
        """
        iNaturalist Dataset generated from a .pkl containing a list of tuples
         each of them representing a path to an image and it class
        :param pickle_file: path to .pkl file
        :param root_path: path to the directory containing images
        :param device:
        :param transforms: list of transformation to apply to images
        """
        self.root_path = root_path
        self.device = device
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.transforms = transforms

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = Image.open(os.path.join(self.root_path, img_path)).convert("RGB")
        label = torch.tensor(label).to(self.device)

        if self.transforms:
            img = self.transforms(img).to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)


def get_iterator_inaturalist(file_path, device, batch_size=1):
    """
    returns an iterator over iNaturalist dataset batches
    :param file_path: path to .pkl file containing a list of tuples
         each of them representing a path to an image and it class
    :param device:
    :param batch_size:
    :return: torch.utils.DataLoader object constructed from INaturalist dataset object
    """
    root_path = os.path.join("data", "inaturalist")

    transforms = Compose([CenterCrop((224, 224)),
                          ToTensor(),
                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                          ])

    dataset = INaturalist(file_path, device=device, root_path=root_path, transforms=transforms)
    iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return iterator
