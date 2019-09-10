
import pandas as pd
import rawpy
import torchvision.transforms as transforms
import torch
import os
import numpy as np


# transform_image = transforms.Compose([transforms.Resize((512, 512)),
#                                       transforms.ToTensor()])

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


class Preprocess:
    def __init__(self, path_txt):
        dataframe = pd.read_csv(path_txt, sep=' ', header=None)
        data = dataframe.values

        input = data[:, 0].flatten()
        target = data[:, 1].flatten()

        first = []
        second = []

        for index in range(len(input)):
            _, train_fn = os.path.split(input[index])
            first.append(train_fn)
            _, target_fn = os.path.split(target[index])
            second.append(target_fn)

        self.data = first
        self.target = second

    def transform_to_tensor(self, path):
        last_path_target = ''

        input_tensor_txt = []
        target_tensor_txt = []

        for index in range(len(self.data)):
            print(index)
            raw_input = rawpy.imread(path + '/short/' + self.data[index])
            # im_input = raw_input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

            input_tensor_txt.append(self.data[index])
            target_tensor_txt.append(self.target[index])

            input_exposure = float(self.data[index][9:-5])
            target_exposure = float(self.target[index][9:-5])
            ratio = min(target_exposure / input_exposure, 300)
            im = np.expand_dims(pack_raw(raw_input), axis=0) * ratio

            torch.save(im,
                       '/media/student/SPATIU/LTSID/TensorImage/Input/' +
                       self.data[index][0:-4])

            if (last_path_target != self.target[index]):
                raw_target = rawpy.imread(path + '/long/' + self.target[index])
                last_path_target = self.target[index]
                im_target = raw_target.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True,
                                                   output_bps=16)
                im_target = np.expand_dims(np.float32(im_target / 65535.0), axis=0)
                torch.save(im_target,
                           '/media/student/SPATIU/LTSID/TensorImage/Target/' +
                           self.target[index][0:-4])

        torch.save(input_tensor_txt,
                   '/media/student/SPATIU/LTSID/TensorImage/' + 'input_val_tensors' + '.txt')
        torch.save(target_tensor_txt,
                   '/media/student/SPATIU/LTSID/TensorImage/' + 'target_val_tensors' + '.txt')


# a = Preprocess('/media/student/SPATIU/LTSID/dataset/Sony_train_list.txt')
# a.transform_to_tensor('/media/student/SPATIU/LTSID/dataset/Sony')

# model = np.array((torch.load('/home/freeaccess/PycharmProjects/Learn to see in the dark/Learn to see in the dark/Tensor Val Image/target_val_tensors.txt')))
# print(model)
# model = model.reshape(-1,1)
# print(model)
#
# torch.save(model,'/home/freeaccess/PycharmProjects/Learn to see in the dark/Learn to see in the dark/Tensor Val Image/' + 'best_val_target_tensors.txt')

model = np.array((torch).load('/media/student/SPATIU/LTSID/TensorImage/input_val_tensors.txt'))
print(model)
model = model.reshape((-1,1))
print(model)

torch.save(model, '/media/student/SPATIU/LTSID/TensorImage/' + 'best_input_val_tensors' + '.txt')
