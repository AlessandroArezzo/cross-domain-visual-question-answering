import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', label_datasetA="A", label_datasetB="B",
                 transform_mode=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.label_datasetA = label_datasetA
        self.label_datasetB = label_datasetB
        if not transform_mode:
            self.files_A = sorted(glob.glob(os.path.join(root, mode, label_datasetA) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, mode, label_datasetB) + '/*.*'))
        else:
            self.files = sorted(glob.glob(os.path.join(root, mode, label_datasetA) + '/*.*'))
        self.transform_mode = transform_mode

    def __getitem__(self, index):
        if not self.transform_mode:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

            if self.unaligned:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

            return {self.label_datasetA: item_A, self.label_datasetB: item_B}
        else:
             item = self.transform(Image.open(self.files[index % len(self.files)]))
             return {self.label_datasetA: item}


    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) if not self.transform_mode else len(self.files)