import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import os
import random
# from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(input, target, patch_size, scale = 1, ix=-1, iy=-1):
    ih, iw, channels = input.shape
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)


    input = input[ix:ix + ip, iy:iy + ip, :]  # [:, ty:ty + tp, tx:tx + tp]
    target = target[ix:ix + ip, iy:iy + ip, :]  # [:, iy:iy + ip, ix:ix + ip]


    return  input, target


def augment(inputs, target, hflip, rot):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot180 = rot and random.random() < 0.5

    def _augment(inputs,target):
        if hflip:
            inputs = inputs[:, ::-1, :]
            target = target[:, ::-1, :]
        if vflip:
            inputs = inputs[::-1, :, :]
            target = target[::-1, :, :]
        if rot180:
            inputs = cv2.rotate(inputs, cv2.ROTATE_180)
            target = cv2.rotate(target, cv2.ROTATE_180)
        return inputs, target

    inputs, target = _augment(inputs, target)

    return inputs, target



# def get_image_hdr(img):
#     img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
#     img = np.round(img/(2**6)).astype(np.uint16)
#     img = img.astype(np.float32)/1023.0
#     return img

def get_image_ldr(img):
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    # if img.shape[1]*img.shape[2] >= 800*800:
    #     img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    w,h = img.shape[0],img.shape[1]
    while w%4!=0:
        w+=1
    while h%4!=0:
        h+=1
    img = cv2.resize(img,(h,w))
    return img


def load_image_train2(group):
    # images = [get_image(img) for img in group]
    # inputs = images[:-1]
    # target = images[-1]
    inputs = get_image_ldr(group[0])
    target = get_image_ldr(group[1])
    # if black_edges_crop == True:
    #     inputs = [indiInput[70:470, :, :] for indiInput in inputs]
    #     target = target[280:1880, :, :]
    #     return inputs, target
    # else:
    return inputs, target


def transform():
    return Compose([
        ToTensor(),
    ])

def BGR2RGB_toTensor(inputs, target):
    inputs = inputs[:, :, [2, 1, 0]]
    target = target[:, :, [2, 1, 0]]
    inputs = torch.from_numpy(np.ascontiguousarray(np.transpose(inputs, (2, 0, 1)))).float()
    target = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()
    return inputs, target

class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

    def __getitem__(self, index):

        inputs, target = load_image_train2(self.image_filenames[index])

        #target = target.resize((inputs.size[0], inputs.size[1]), Image.ANTIALIAS)

        # target = cv2.resize(target,(inputs.shape[1],inputs.shape[0]))
        # target = target[:768, :512]
        # inputs = inputs[:768, :512]

        if self.patch_size!=None:
            inputs, target = get_patch(inputs, target, self.patch_size, self.upscale_factor)


        if self.data_augmentation:
            inputs, target = augment(inputs, target, self.hflip, self.rot)

        if self.transform:
            inputs, target = BGR2RGB_toTensor(inputs, target)


        return {'LQ': inputs, 'GT': target, 'LQ_path': self.image_filenames[index][0], 'GT_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)


# if __name__ == '__main__':
#     output = 'visualize'
#     if not os.path.exists(output):
#         os.mkdir(output)
#     dataset = DatasetFromFolder(4, True, 'dataset/groups.txt', 64, True, True, True)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
#     for i, (inputs, target) in enumerate(dataloader):
#         if i > 10:
#             break
#         if not os.path.exists(os.path.join(output, 'group{}'.format(i))):
#             os.mkdir(os.path.join(output, 'group{}'.format(i)))
#         input0, input1, input2, input3, input4 = inputs[0][0], inputs[0][1], inputs[0][2], inputs[0][3], inputs[0][4]
#         vutils.save_image(input0, os.path.join(output, 'group{}'.format(i), 'input0.png'))
#         vutils.save_image(input1, os.path.join(output, 'group{}'.format(i), 'input1.png'))
#         vutils.save_image(input2, os.path.join(output, 'group{}'.format(i), 'input2.png'))
#         vutils.save_image(input3, os.path.join(output, 'group{}'.format(i), 'input3.png'))
#         vutils.save_image(input4, os.path.join(output, 'group{}'.format(i), 'input4.png'))
#         vutils.save_image(target, os.path.join(output, 'group{}'.format(i), 'target.png'))
