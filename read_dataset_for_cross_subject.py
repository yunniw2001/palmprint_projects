import argparse
import os
import matplotlib.pyplot as plt
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
            # transforms.Normalize(128, 128)
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

def read_txt(text_dir):
    file = open(text_dir,'r')
    lines = file.readlines()
    epoch = int(lines[0])
    per_idx = int(lines[1])
    cur_idx =lines[2][1:-2].split(',')
    for i in range(len(cur_idx)):
        cur_idx[i] = int(cur_idx[i])
    train_loss = lines[3][1:-2].split(',')
    for i in range(len(train_loss)):
        train_loss[i] = float(train_loss[i])
    return epoch, per_idx, cur_idx, train_loss

def make_text_save(text_dir,epoch, pre_idx,tain_idx,train_loss):
    write_file = open(text_dir, 'w')
    write_file.write(str(epoch)+'\n')
    write_file.write(str(pre_idx )+ '\n')
    write_file.write(str(tain_idx)+'\n')
    write_file.write(str(train_loss)+'\n')
    write_file.close()
prepare_transform_for_image()

batch_size = 55
num_class = 480
parser = argparse.ArgumentParser(description='my argument')
args = parser.parse_known_args()[0]
args.lr = 0.01
args.num_classes = num_class
args.alpha = args.lr
train_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\train',
                          'E:\digital_image_processing\data_1\\tongji_cross_subject\\train_label.txt',
                          preprocessing)
test_dataset = MyDataset('E:\digital_image_processing\data_1\\tongji_cross_subject\\test',
                         'E:\digital_image_processing\data_1\\tongji_cross_subject\\test_label.txt', preprocessing)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2, pin_memory=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers = 2, pin_memory=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义网络
net = ResNet.ResNet20(num_class,s=30,m=0.65,lamda=0.1)
net.to(device)
flag = True
PATH = "update_model459.pt"
param = torch.load(PATH,map_location=torch.device('cpu'))
net.load_state_dict(param)

optimizer = optim.SGD(net.parameters(), lr=args.lr,weight_decay=0.0005)

epochs = 3000
torch.autograd.set_grad_enabled(True)
# 训练
train_loss = []
train_idx = []
per_idx = 0
train_accuracy = []
epoch = 0
while epoch < epochs:
    running_loss = 0.0
    if flag == True:
        epoch, per_idx, train_idx, train_loss = read_txt('data.txt')
        flag = False
    for i, data in enumerate(train_dataloader):
        images, label = data
        prec, loss = net(images.to(device), label.to(device), args)
        # print(net.C_LMCL.weight)
        # print('======')
        # print(net.C_LMCL.weight.data)
        running_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print('[epoch:%d  %5d]   loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            print(prec)
            train_loss.append(running_loss / 50)
            train_idx.append(per_idx)
            per_idx += 1
            running_loss = 0
    if (epoch + 1) % 20 == 0:
        save_path = '/home/ubuntu/project/data.txt'
        make_text_save(save_path, epoch, per_idx, train_idx, train_loss)
        PATH = "/home/ubuntu/project/update_model" + str(epoch) + ".pt"
        # Save
        torch.save(net.state_dict(), PATH)
    epoch += 1
# 测试- 掌纹配对
for i, data in enumerate(train_dataloader):
    images, label = data
    # for item in images:
    #     print(item)
    # break
    prec,loss = net(images.to(device),label.to(device),args)



