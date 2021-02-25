import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

Image.MAX_IMAGE_PIXELS = 758520000

class ImageDataset(Dataset):
    def __init__(self, pathA, pathB=[], transforms_=None, unaligned=False, mode='train', label_datasetA="A",
                 label_datasetB="B", transform_mode=False, percent_trainA=None, percent_trainB=None):
        assert mode == 'train' or mode == 'test'
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.label_datasetA = label_datasetA
        self.label_datasetB = label_datasetB
        self.files_A = self.read_images_from_path(pathA, mode, percent_trainA)
        if not transform_mode:
            self.files_B = self.read_images_from_path(pathB, mode, percent_trainB)
        self.transform_mode = transform_mode

    def __getitem__(self, index):
        if not self.transform_mode:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            if self.unaligned:
                image = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB')
                item_B = self.transform(image)
                #item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

            return {self.label_datasetA: item_A, self.label_datasetB: item_B}
        else:
             item = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
             return {self.label_datasetA: item}

    def read_images_from_path(self, path, mode, percent_train):
        files_to_add = []
        for dir in path:
            if percent_train == None:
                files_to_add += sorted(glob.glob(os.path.join(dir, mode) + '/*.*'))
            else:
                files = sorted(glob.glob(os.path.join(dir) + '/*.*'))
                if mode == 'train':
                    files_to_add += files[:int(len(files) * percent_train / 100)]
                else:
                    files_to_add += files[int(len(files) * (100 - percent_train) / 100):]
        return files_to_add

    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) if not self.transform_mode else len(self.files_A)