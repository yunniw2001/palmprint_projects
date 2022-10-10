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
import ResNet
import C_LMCL_Loss
import torch.optim as optim

preprocessing = None


def prepare_transform_for_image():
    global preprocessing
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((224, 224))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=50, contrast=0.5)
    smooth_or_sharpening = transforms.RandomChoice([
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
            transforms.RandomApply(
                [rotation, contrast_brightness_adjustment, smooth_or_sharpening, color_shift], 0.6),
            resized_cropping,
            transforms.ToTensor(),
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


def update_some_center_vector(N, feature, weight, label, j, alpha):
    sum_ch = 0
    sum_p = 0
    for i in range(N):
        if label[i] == j:
            sum_ch += (weight[j] - feature[i])
            sum_p += 1
    delta = sum_ch / (1 + sum_p)
    return torch.sub(weight[j], torch.mul(alpha, delta))


prepare_transform_for_image()

batch_size = 55
num_class = 300
lr = 0.01
train_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\mini_train',
                          'E:\digital_image_processing\data_1\\tongji_cross_subject\\mini_label.txt',
                          preprocessing)
test_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\test',
                         'E:\digital_image_processing\data_1\\tongji_cross_subject\\test_label.txt', preprocessing)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# 定义网络
net = ResNet.ResNet20()
for param in net.C_LMCL.parameters():
    param.requires_grad = False

criterion = C_LMCL_Loss.CLMCLLoss(batch_size=batch_size, class_num=num_class, m=0.65, s=30, alpha=lr, lamda=0.1)
optimizer = optim.SGD(net.parameters(), lr=lr)

epochs = 10
torch.autograd.set_grad_enabled(True)
# 训练
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        images, label = data
        feature = net(images)[0]
        # print(net.C_LMCL.weight)
        # print('======')
        # print(net.C_LMCL.weight.data)
        loss = criterion(feature, net.C_LMCL.weight, label)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            N = batch_size
            if batch_size > len((label)):
                N = len(label)
            for j in range(num_class):
                net.C_LMCL.weight.data[j] = update_some_center_vector(N, feature, net.C_LMCL.weight.data[j], label, j, lr)
        net.C_LMCL.grad.zero_()
        print('[%d  %5d]   loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0
        sys.exit(0)
