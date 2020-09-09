import cv2
from torch.utils.data import Dataset
from os.path import join
from os import listdir
import torch

class HuskyWolfDataset(Dataset):
    def __init__(self, folders, transform=None):
        self.examples = []
        self.transform = transform

        for idx, folder in enumerate(folders):
            files = [(join(folder, file), idx) for file in listdir(folder)]
            self.examples += files

    def __getitem__(self, item):
        path, label = self.examples[item]
        image = cv2.imread(path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.examples)


