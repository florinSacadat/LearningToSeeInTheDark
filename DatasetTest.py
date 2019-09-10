import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SID(Dataset):
    def __init__(self, input_txt_path, target_txt_path, input_images, target_images, scale_images):
        self.input = np.array(torch.load(input_txt_path)).flatten()
        self.target = np.array(torch.load(target_txt_path)).flatten()
        self.input_images = input_images
        self.target_images = target_images
        self.scale_images= scale_images


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = torch.load(self.input_images + self.input[idx][0:-4])
        target = torch.load(self.target_images + self.target[idx][0:-4])
        scale = torch.load(self.scale_images + self.input[idx][0:-4])


        input_patch = input.squeeze()
        target_patch = target.squeeze()
        scale_patch = scale.squeeze()

        return input_patch, target_patch, scale_patch