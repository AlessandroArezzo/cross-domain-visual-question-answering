import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError, ImageFile
import torchvision.transforms as transforms
from random import shuffle
from pathlib import Path

Image.MAX_IMAGE_PIXELS = 2553880864
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, pathA, pathB=[], transforms_=None, unaligned=False, mode='train', label_datasetA="A",
                 label_datasetB="B", transform_mode=False, percent_trainA=None, percent_trainB=None, shuffle=True,
                 existing_path=None):
        assert mode == 'train' or mode == 'test'
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.label_datasetA = label_datasetA
        self.label_datasetB = label_datasetB
        self.shuffle = shuffle
        self.files_A, self.filesname_A = self.__read_images_from_path(pathA, mode, percent_trainA, transform_mode,
                                                                      existing_path)
        if not transform_mode:
            self.files_B, _ = self.__read_images_from_path(pathB, mode, percent_trainB)
        self.transform_mode = transform_mode

    def __getitem__(self, index):
        if not self.transform_mode:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
            if self.unaligned:
                while True:
                    try:
                        item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
                        break
                    except OSError:
                        continue
            else:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

            return {self.label_datasetA: item_A, self.label_datasetB: item_B}
        else:
             item = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))
             return {self.label_datasetA: item, "name": self.filesname_A[index % len(self.filesname_A)]}

    def __read_images_from_path(self, path, mode, percent_train, transform_mode=False, existing_path=None):
        files_to_add = []
        for dir in path:
            if percent_train == None and not transform_mode:
                files_to_add += sorted(glob.glob(os.path.join(dir, mode) + '/*.*'))
            elif percent_train == None and transform_mode:
                files_to_add += sorted(glob.glob(dir + '/*.*'))
            else:
                files = glob.glob(os.path.join(dir) + '/*.*')
                if self.shuffle:
                    shuffle(files)
                if mode == 'train':
                    files_to_add += files[:int(len(files) * percent_train / 100)]
                else:
                    files_to_add += files[int(len(files) * (100 - percent_train) / 100):]
        return self.__check_images_consistency(files_to_add, existing_path)

    def __check_images_consistency(self, files, existing_path=None):
        imgs = []
        imgs_name = []
        for img_file in files:
            try:
                if not existing_path or not os.path.exists(os.path.join(existing_path,
                                                                        '%04d'%int(Path(img_file).stem)+".png")):
                    Image.open(img_file)
                    imgs.append(img_file)
                    imgs_name.append(Path(img_file).stem)
            except UnidentifiedImageError:
                continue
            except OSError:
                continue
        return imgs, imgs_name


    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) if not self.transform_mode else len(self.files_A)