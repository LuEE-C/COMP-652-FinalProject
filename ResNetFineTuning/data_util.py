from torchvision import transforms, utils
from torch.utils.data import Dataset
import torch
from skimage import io
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


class TrainProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, oversampling=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
        train = Counter()
        for i, ii in self.prots_df.iterrows():
            train.update(ii['Target'].split())
        good_val = False
        idx = 0
        while good_val is False:
            idx += 1
            good_val = True
            self.prots_df = self.prots_df.sample(frac=1)
            self.validation_df = self.prots_df[int(.8 * len(self.prots_df)):]
            val = Counter()
            for i, ii in self.validation_df.iterrows():
                val.update(ii['Target'].split())

            for i in range(28):
                if val[str(i)] == 0:
                    good_val = False
                val_ratio = val[str(i)]/len(self.validation_df)
                train_ratio = train[str(i)] / int(.8 * len(self.prots_df))
                if ((1/1.5) < val_ratio/train_ratio < 1.5) is False:
                    good_val = False

        self.validation_df.to_csv(os.path.join(root_dir, 'validation.csv'), index=False)
        self.prots_df = self.prots_df[:int(.8 * len(self.prots_df))]

        self.oversampling(oversampling)

        self.targets = torch.zeros(size=(len(self.prots_df), 28), dtype=torch.float)
        for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: list(map(int, x.split()))).values.tolist()):
            for cat in cats:
                self.targets[i, cat] = 1

        self.root_dir = os.path.join(root_dir, 'train_data')
        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

    def oversampling(self, ratio=100):
        c = Counter()
        for i, ii in self.prots_df.iterrows():
            c.update(ii['Target'].split())

        self.ratios = np.array([c[str(i)]/len(self.prots_df) for i in range(28)])

        max_v = 0
        for i in c:
            if c[i] > max_v:
                max_v = c[i]
        max_v = max_v/ratio
        extra_x, extra_y = [], []
        for value in range(28):

            oversample = int(max_v * (1 - (c[str(value)] / max_v)))
            relevant_indexes = []
            for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: x.split()).values.tolist()):
                if str(value) in cats:
                    relevant_indexes.append(i)
            indexes = [random.choice(relevant_indexes) for _ in range(oversample)]

            for idx in indexes:
                extra_x.append(self.prots_df['Id'].values[idx])
                extra_y.append(self.prots_df['Target'].values[idx])

        x = np.concatenate((self.prots_df['Id'].values, extra_x), axis=0)
        y = np.concatenate((self.prots_df['Target'], extra_y), axis=0)
        self.prots_df = pd.DataFrame(columns=['Id', 'Target'])
        self.prots_df['Id'] = x
        self.prots_df['Target'] = y

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.prots_df.iloc[idx, 0])
        img_labels = self.targets[idx]
        img = self.transform(Image.fromarray(
            np.swapaxes(np.array([io.imread(img_name + '_' + color + '.png') for color in self.color], dtype=np.uint8),
                        0, 2)))
        return img, img_labels


class ValProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv(os.path.join(root_dir, 'validation.csv'))
        self.targets = torch.zeros(size=(len(self.prots_df), 28), dtype=torch.float)
        for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: list(map(int, x.split()))).values.tolist()):
            for cat in cats:
                self.targets[i, cat] = 1

        self.root_dir = os.path.join(root_dir, 'train_data')
        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.prots_df.iloc[idx, 0])
        img_labels = self.targets[idx]
        img = self.transform(Image.fromarray(
            np.swapaxes(np.array([io.imread(img_name + '_' + color + '.png') for color in self.color], dtype=np.uint8),
                        0, 2)))

        return img, img_labels


class TestProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))
        self.root_dir = os.path.join(root_dir, 'test_data')
        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.prots_df.iloc[idx, 0])
        img = self.transform(Image.fromarray(
            np.swapaxes(np.array([io.imread(img_name + '_' + color + '.png') for color in self.color], dtype=np.uint8),
                        0, 2)))
        return img


if __name__ == '__main__':
    mean = [0.5, 0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5, 0.5]
    scale = 360
    input_shape = 299
    transf = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    p = TrainProtsDataset('../prots_data/', transform=transf, oversampling=True)
    # print(p.prots_df.head())
    # img, img_lab = p[0]
    # print(img.shape)
    #
    # dataloader = DataLoader(p, batch_size=16,
    #                         shuffle=True, num_workers=4)
