import albumentations as A

from os.path import join, dirname, realpath
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lime_image.dataset import HuskyWolfDataset


class HuskyWolfDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_transform, val_transform):
        super().__init__()
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage=None):
        transform = A.Compose([
                    A.Resize(256, 256),
                    A.RandomCrop(224, 224),
                    A.HorizontalFlip(),
                    A.RandomRotate90(),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                    ])
        husky_train_path = join(dirname(realpath(__file__)), "husky_train")
        wolf_train_path = join(dirname(realpath(__file__)), "wolf_train")

        husky_test_path = join(dirname(realpath(__file__)), "husky_test")
        wolf_test_path = join(dirname(realpath(__file__)), "wolf_test")

        self.train_set = HuskyWolfDataset([husky_train_path, wolf_train_path], self.train_transform)
        self.val_set = HuskyWolfDataset([husky_test_path, wolf_test_path], self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
