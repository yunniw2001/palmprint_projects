import math

import numpy as np
import torch.nn
import torch.nn as nn


class CLMCLLoss(nn.Module):
    def __init__(self, batch_size, class_num, m, s, alpha, lamda):
        super(CLMCLLoss, self).__init__()
        self.N = batch_size
        self.batch_size = batch_size
        self.class_num = class_num
        self.m = m
        self.s = s
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, feature, weight, label):
        if self.batch_size > len(label):
            self.N = len(label)
        loss_lmc = self.calculate_lmc_loss(feature, weight, label)
        loss_c = self.calculate_c_loss(feature, weight, label)
        loss_c_lmcl = loss_lmc + self.lamda * loss_c
        # for j in range(self.class_num):
        #     weight[j] = self.update_some_center_vector(feature, weight, label, j)
        return loss_c_lmcl

    def calculate_lmc_loss(self, feature, weight, label):
        tmp_loss = 0
        for i in range(self.N):
            w_yi = weight[label[i]]
            cos_theta_yi = self.calculate_cos_theta_j(w_yi, feature[i])
            cur_res = torch.exp(self.s * (cos_theta_yi - self.m)) / (
                        torch.exp(self.s * (cos_theta_yi - self.m)) + self.calculate_sum(feature, weight, i, label[i]))
            tmp_loss += torch.log(cur_res)
            # print('s',end='')
            # print(s)
        return -tmp_loss / self.N

    def calculate_cos_theta_j(self, w_j, x_i):
        # print(np.linalg.norm(w_j))
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        return cos_sim(w_j,x_i)

    def calculate_sum(self, feature, weight, i, y_i):
        sum = 0
        for j in range(self.class_num):
            if j != y_i:
                w_j = weight[j]
                sum += torch.exp(self.s * self.calculate_cos_theta_j(w_j, feature[i]))
        return sum

    def calculate_c_loss(self, feature, weight, label):
        ans = 0
        for i in range(self.N):
            ans += (torch.norm(feature[i] - weight[label[i]]) ** 2)
        return ans / 2

    def update_some_center_vector(self, feature, weight, label, j):
        sum_ch = 0
        sum_p = 0
        for i in range(self.N):
            if label[i] == j:
                sum_ch += (weight[j] - feature[i])
                sum_p += 1
        delta = sum_ch / (1 + sum_p)
        return torch.sub(weight[j],torch.mul(self.alpha,delta))