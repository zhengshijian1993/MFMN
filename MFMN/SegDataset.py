import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import cv2 as cv

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.dir_A = os.path.join(self.root, self.phase + '/LQ')
        self.dir_B = os.path.join(self.root, self.phase + '/GT')
        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")

        # resize
        resized_a = A.resize((280, 280), Image.LANCZOS)
        resized_b = B.resize((280, 280), Image.LANCZOS)

        # crop the training image into fineSize
        w, h = resized_a.size
        x, y = randrange(w - self.fineSize + 1), randrange(h - self.fineSize + 1)
        cropped_a = resized_a.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_b = resized_b.crop((x, y, x + self.fineSize, y + self.fineSize))


        # transform to (0, 1)
        tensor_a = self.transform(cropped_a)
        tensor_b = self.transform(cropped_b)


        return tensor_a, tensor_b

    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot, phase):
        super().__init__()
        self.root = dataroot
        self.phase = phase

        self.dir_A = os.path.join(self.root, self.phase + '/LQ')
          # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        A = Image.open(self.A_paths[index]).convert("RGB")
        name = self.A_paths[index].split("/")[-1]
        # name =name.split(".")[0]
        name = os.path.splitext(name)[0]
        # resize

        tensor_a = self.transform(A)
        # tensor_b = self.transform(B)

        return tensor_a, name

    def __len__(self):
        return len(self.A_paths)