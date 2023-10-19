import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from torchvision.utils import save_image

#######################################################
class feature_vision(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out, name):
        [c, h, w] = out[0].shape

        ans = np.zeros((h, w))
        dst = './features'
        therd_size = 256  # 有些图太小，会放大到这个尺寸
        dst_path = os.path.join(dst)
        # os.makedirs(dst_path)
        ret = []
        y = np.asarray(out[0].data.cpu())  # 处理成array格式
        for i in range(c):
            y_ = y[i, :, :]
            ret.append(np.mean(y_))#把每个feature map的均值作为对应权重
        for j in range(h):
            for k in range(w):
                for i in range(c):
                    ans[j][k] += ret[i] * y[i][j][k]#加权融合
        for i in range(h):
            for j in range(w):
                ans[i][j] = max(0, ans[i][j])#不需要负值
        ans = (ans - np.min(ans)) / np.max(ans)#归一化
        print('ans\n', ans)
        ans = np.asarray(ans * 255, dtype=np.uint8)  # [0,255]
        print('ans _type:', ans.dtype)  # uint8
        # y_ = cv2.applyColorMap(y_, cv2.COLORMAP_JET)
        ans = cv2.applyColorMap(ans, cv2.COLORMAP_JET)  # https://www.sohu.com/a/343215045_120197868创建伪彩色
        if ans.shape[0] < therd_size:  # 大小调整
            tmp_file = os.path.join(dst_path, str(therd_size) + name + 'sr.png')
            tmp_img = ans.copy()  # 将src中的元素复制到tensor
            tmp_img = cv2.resize(tmp_img, (therd_size, therd_size),
                                 interpolation=cv2.INTER_NEAREST)  # https://www.cnblogs.com/jyxbk/p/7651241.html
            cv2.imwrite(tmp_file, tmp_img)
        dst_file = os.path.join(dst_path, name + 'sr.png')
        cv2.imwrite(dst_file, ans)
        return out

class MSM(nn.Module):
    def __init__(self, in_channels):
        super(MSM, self).__init__()

        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels, in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_2 = nn.Conv2d(in_channels//8, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_3 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_4 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input):

        x1 = self.avg_pool(input)
        x1 = self.conv_1(x1)
        x1 = self.act(x1)
        x1 = self.conv_2(x1)

        x2 = self.max_pool(input)
        x2 = self.conv_1(x2)
        x2 = self.act(x2)
        x2 = self.conv_2(x2)

        mid = x1 + x2
        mid = self.act(mid)

        out = mid * input

        return out


class SCFM(nn.Module):
    def __init__(self, in_channels):
        super(SCFM, self).__init__()


        self.conv_first_r = nn.Conv2d(in_channels, in_channels//8, kernel_size=3, stride=1, padding=1, bias=False,groups=2)
        self.conv_first_r2 = nn.Conv2d(in_channels, in_channels//8, kernel_size=3, stride=1, padding=1, bias=False,groups=2)
        self.conv_first_r3 = nn.Conv2d(in_channels, in_channels//8, kernel_size=3, stride=1, padding=1, bias=False,groups=2)
        self.conv_first_r4 = nn.Conv2d(in_channels, in_channels//8, kernel_size=3, stride=1, padding=1, bias=False,groups=2)


        self.conv0 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3,padding=1,groups=4)
        self.conv1 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels // 2, kernel_size=3, padding=1, groups=4)



        self.conv_h = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_r = nn.InstanceNorm2d(in_channels//2, affine=True)


        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.ReLU()

        self.conv_feature = nn.Conv2d(in_channels // 2, 3, kernel_size=1, padding=0)

        self.conv_feature1 = nn.Conv2d(in_channels // 2, 3, kernel_size=1, padding=0)

        self.fea = feature_vision()

    def forward(self, input):

        x1 = self.conv_first_r(input)
        x2 = self.conv_first_r2(input)
        x3 = self.conv_first_r3(input)
        x4 = self.conv_first_r4(input)

        x_1 = torch.cat((x1,x2,x3,x4),dim=1)

        x_1 = self.conv0(x_1)
        x_1 = self.conv1(x_1)
        x_1 = self.conv2(x_1)

        # s1 = self.conv_feature(x_1)
        # save_image(s1, "features/2-1.png", nrow=1, normalize=False)
        self.fea(x_1, "2-1")

        out_skip = self.conv_h(input)
        out_instance_r = self.instance_r(out_skip)

        self.fea(out_instance_r, "2-2")
        # s2 = self.conv_feature1(out_instance_r)
        # save_image(s2, "features/2-2.png", nrow=1, normalize=False)

        out_instance = torch.cat((x_1, out_instance_r), dim=1)

        out_instance = self.conv_2(out_instance)
        out_instance = self.act(out_instance)

        return out_instance



################################################################
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class MMM_block(nn.Module):
    def __init__(self, input):
        super(MMM_block, self).__init__()
        self.LN = LayerNorm(input, data_format='channels_first')
        self.SCFM = SCFM(input)
        self.MFM = MSM(input)

        self.conv_feature1 = nn.Conv2d(input, 3, kernel_size=1, padding=0)
        self.conv_feature2 = nn.Conv2d(input, 3, kernel_size=1, padding=0)
        self.conv_feature3 = nn.Conv2d(input, 3, kernel_size=1, padding=0)
        self.conv_feature4 = nn.Conv2d(input, 3, kernel_size=1, padding=0)
        self.conv_feature5 = nn.Conv2d(input, 3, kernel_size=1, padding=0)
        self.conv_feature6 = nn.Conv2d(input, 3, kernel_size=1, padding=0)

        self.fea = feature_vision()

    def forward(self, x):
        out1 = self.LN(x)

        self.fea(out1, "1")
        # s3 = self.conv_feature1(out1)
        # save_image(s3, "features/1.png", nrow=1, normalize=False)


        out2 = self.SCFM(out1)
        self.fea(out2, "2")
        # s4 = self.conv_feature2(out2)
        # save_image(s4, "features/2.png", nrow=1, normalize=False)


        out3 = out2 + x
        self.fea(out3, "3")
        # s5 = self.conv_feature3(out3)
        # save_image(s5, "features/3.png", nrow=1, normalize=False)


        out4 = self.LN(out3)
        self.fea(out4, "4")
        # s6 = self.conv_feature4(out4)
        # save_image(s6, "features/4.png", nrow=1, normalize=False)


        out5 = self.MFM(out4)
        self.fea(out5, "5")
        # s7 = self.conv_feature5(out5)
        # save_image(s7, "features/5.png", nrow=1, normalize=False)


        out = out5 + out3
        self.fea(out, "6")
        # s8 = self.conv_feature6(out)
        # save_image(s8, "features/6.png", nrow=1, normalize=False)

        return out


class Encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64):           # 16/64/128
        super(Encoder, self).__init__()
        self.num_block = 11
        self.model = nn.Sequential(*[MMM_block(base_nf) for _ in range(self.num_block)])

        self.model1 = MMM_block(base_nf)

        self.model2 = nn.Sequential(*[MMM_block(base_nf) for _ in range(self.num_block)])

        self.model3 = MMM_block(base_nf)
        self.model4 = MMM_block(base_nf)


        self.conv1 = nn.Conv2d(in_nc, base_nf , 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)                             # head
        res = out

        x1 = self.model1(res)               #1
        x12 = self.model(x1)                #12

        x23 =  self.model2(x12)            #23
        x24 = self.model4(x23)            #24

        out_1 = x24 + out                        # body

        out = self.conv3(self.act(out_1))                          # tail

        return out

