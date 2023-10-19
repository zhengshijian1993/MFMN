import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn
from torchvision import models
import math
class VGG_loss(nn.Module):
    def __init__(self, model):
        super(VGG_loss, self).__init__()
        self.features = nn.Sequential(*list(model.children())[0][:-3])
        self.l1loss = nn.L1Loss()
    def forward(self, x,y):
        x_vgg=self.features(x)
        y_vgg=self.features(y)
        loss=self.l1loss(x_vgg, y_vgg)
        return loss



class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        L1_temp = 0.8 * self.L1(xs, ys)
        L_total = L1_temp + L2_temp
        return L_total


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, res, gt):
        res = (res + 1.0) * 127.5
        gt = (gt + 1.0) * 127.5
        r_mean = (res[:, 0, :, :] + gt[:, 0, :, :]) / 2.0
        r = res[:, 0, :, :] - gt[:, 0, :, :]
        g = res[:, 1, :, :] - gt[:, 1, :, :]
        b = res[:, 2, :, :] - gt[:, 2, :, :]
        p_loss_temp = (((512 + r_mean) * r * r) / 256) + 4 * g * g + (((767 - r_mean) * b * b) / 256)
        p_loss = torch.mean(torch.sqrt(p_loss_temp + 1e-8)) / 255.0
        return p_loss



###################################################################
#
#                     Retinex_loss
#
##################################################################

r = 0
s = [15, 60, 90]


class MyGaussianBlur(torch.nn.Module):
    # 初始化
    def __init__(self, radius=1, sigema=1.5):
        super(MyGaussianBlur, self).__init__()
        self.radius = radius
        self.sigema = sigema

    # 高斯的计算公式
    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2

    # 滤波模板
    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(0, sideLength):
            for j in range(0, sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all

    # 滤波函数
    def filter(self, image, template):
        kernel = torch.FloatTensor(template).cuda()
        kernel2 = kernel.expand(3, 1, 2 * r + 1, 2 * r + 1)
        weight = torch.nn.Parameter(data=kernel2, requires_grad=False)
        new_pic2 = torch.nn.functional.conv2d(image, weight, padding=r, groups=3)
        return new_pic2


# print(loss.item())
def MutiScaleLuminanceEstimation(img):
    guas_15 = MyGaussianBlur(radius=r, sigema=15).cuda()
    temp_15 = guas_15.template()

    guas_60 = MyGaussianBlur(radius=r, sigema=60).cuda()
    temp_60 = guas_60.template()

    guas_90 = MyGaussianBlur(radius=r, sigema=90).cuda()
    temp_90 = guas_90.template()
    x_15 = guas_15.filter(img, temp_15)
    x_60 = guas_60.filter(img, temp_60)
    x_90 = guas_90.filter(img, temp_90)
    img = (x_15 + x_60 + x_90) / 3

    return img


class Retinex_loss1(nn.Module):
    def __init__(self):
        super(Retinex_loss1, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, y, x):
        batch_size, h_x, w_x = x.size()[0], x.size()[2], x.size()[3]
        x = MutiScaleLuminanceEstimation(x)
        y = MutiScaleLuminanceEstimation(y)

        retinex_loss_L1 = self.L1(x, y)

        return retinex_loss_L1



class Combinedloss(torch.nn.Module):
    def __init__(self):
        super(Combinedloss, self).__init__()
        self.loss_1 = MyLoss()
        vgg = models.vgg19_bn(pretrained=True)
        self.vggloss = VGG_loss(vgg)
        self.loss_2 = Retinex_loss1()
        self.loss_3 = ColorLoss()


    def forward(self, out, label):
        l1_loss = self.loss_1(out, label)
        loss_2 = self.loss_2(out, label)
        # loss_3 = self.loss_3(out, label)
        vgg_loss = self.vggloss(out, label)


        total_loss = l1_loss + vgg_loss + loss_2

        return total_loss