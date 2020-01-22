import torch
from torch.utils.data import Dataset

from utils import preprocess_image, get_mask_and_regr, imread
import numpy as np


class CarDataset(Dataset):
    """Car dataset."""

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

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Read image
        img0 = imread(img_name, True, self.training)
        img = preprocess_image(img0)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        if self.training:
            mask, regr = get_mask_and_regr(img0, labels)
            regr = np.rollaxis(regr, 2, 0)
        else:
            mask, regr = 0, 0

        return [img, mask, regr]
