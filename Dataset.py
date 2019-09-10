import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SeeInDark(Dataset):
    def __init__(self, input_txt_path, target_txt_path, input_images, target_images):
        self.input = np.array(torch.load(input_txt_path)).flatten()
        self.target = np.array(torch.load(target_txt_path)).flatten()
        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = torch.load(self.input_images + self.input[idx][0:-4])
        target = torch.load(self.target_images + self.target[idx][0:-4])


        input_patch = input.squeeze()
        target_patch = target.squeeze()

        # crop
        ps = 512
        H = input_patch.shape[0]
        W = input_patch.shape[1]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_patch[yy:yy + ps, xx:xx + ps, :]
        target_patch = target_patch[yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        # random flip
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            target_patch = np.flip(target_patch, axis=1).copy()
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0).copy()
            target_patch = np.flip(target_patch, axis=0).copy()

        return input_patch, target_patch