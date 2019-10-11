from utils import *
from torch.utils.data import Dataset
import torch
import numpy as np

class PapirusDataset(Dataset):
    """Papirus dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        image = load_image(img_name)
        if self.training:
            mask = get_mask(image, labels)
        shape = np.array(image.shape)[:2]

        image = to_square(image)
        image = preprocess(image, shape[0], shape[1])
        if self.training:
            mask = to_square(mask)

        image = np.rollaxis(image, 2, 0)
        if self.training:
            mask = np.rollaxis(mask, 2, 0)
        else:
            mask = 0

        sample = [image, mask, shape]

        if self.transform:
            sample = self.transform(sample)

        return sample


class TestDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.df.values[idx][0]
        img_name = self.root_dir.format(idx)

        image = load_image(img_name)

        shape = np.array(image.shape)[:2]

        image = to_square(image)
        image = preprocess(image, shape[0], shape[1])

        image = np.rollaxis(image, 2, 0)

        sample = [image, shape]

        if self.transform:
            sample = self.transform(sample)

        return sample
