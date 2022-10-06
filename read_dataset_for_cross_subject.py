import os
import random
import sys

import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import mytransform

preprocessing = None
train_dataset = None
test_dataset = None
train_dataloader = None
test_dataloader = None


def prepare_transform_for_image():
    global preprocessing
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((224, 224))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=50, contrast=0.5)
    smooth_or_sharpening = transforms.Compose([
        mytransform.MeanFiltersTransform(),
        mytransform.MedianFiltersTransform(),
        mytransform.GaussFiltersTransform(),
        mytransform.GaussianFiltersTransformUnsharpMask(),
        mytransform.MedianFiltersTransformUnsharpMask(),
        mytransform.MeanFiltersTransformUnsharpMask()
    ])
    color_shift = transforms.ColorJitter(hue=0.14)
    preprocessing = transforms.Compose(
        [
            transforms.RandomApply(rotation, 0.6),
            transforms.RandomApply(resized_cropping, 0.6),
            transforms.RandomApply(contrast_brightness_adjustment, 0.6),
            transforms.RandomApply(transforms.RandomChoice(smooth_or_sharpening), 0.6),
            transforms.RandomApply(color_shift, 0.6),
            transforms.Normalize(128, 128)
        ]
    )


def rename_different_session(path):
    filelist = os.listdir(path)
    total_num = len(filelist)
    for item in filelist:
        src = os.path.join(os.path.abspath(path), item)
        ori_name = list(item)
        ori_name[0] = '1'
        ori_name = ''.join(ori_name)
        dst = os.path.join(os.path.abspath(path), ori_name)
        print(dst)
        os.rename(src, dst)
    # print(filelist)


def make_text(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'w')

    for item in files:
        tmp_item = list(item[0:5])
        if tmp_item[0] == '1':
            tmp_item[0] = '0'
        tmp_item = ''.join(tmp_item)
        tmp_item = int(tmp_item)
        belong_index = (tmp_item - 1) // 20
        write_file.write(item + ' ' + str(belong_index) + '\n')
    write_file.close()


class MyDataset(Dataset):
    def __init__(self, img_path, txt_path, transform=None):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.txt_path = txt_path
        f = open(self.txt_path, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            word = line.rstrip().split(' ')
            # print(word)
            imgs.append(os.path.join(self.img_path, word[0]))
            # print(imgs)
            labels.append(word[1])
            # print(labels)
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')
        if transforms is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label


def prepare_dataset():
    global train_dataset, test_dataset, train_dataloader, test_dataloader

    train_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\train',
                              'E:\digital_image_processing\data_1\\tongji_cross_subject\\train_label.txt',
                              preprocessing)
    test_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\test',
                             'E:\digital_image_processing\data_1\\tongji_cross_subject\\test_label.txt', preprocessing)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


if __name__ == '__main__':
    # rename_different_session('E:\digital_image_processing\datasets\\tongji\ROI\session2_rename\\')
    # make_text('E:\digital_image_processing\data_1\\tongji_cross_subject\\train','E:\digital_image_processing\data_1\\tongji_cross_subject\\train_label.txt')
    # make_text('E:\digital_image_processing\data_1\\tongji_cross_subject\\test',
    #           'E:\digital_image_processing\data_1\\tongji_cross_subject\\test_label.txt')
    sys.exit(0)
