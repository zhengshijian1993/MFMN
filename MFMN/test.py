from PIL import Image
import os
import numpy as np
import torch

from NEW_ARCH import Encoder
from SegDataset import TestData
from tqdm import trange
from torchvision.utils import save_image
import torch.nn as nn
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Encoder()

model.load_state_dict(torch.load('logs/1.pth'))

model.eval()
model = model.to(device)

# test, path_list_images_test= read_file_list( type='test')
data_test = TestData("/home/ty/data/zsj/1/USLN-master/features", "UIE_TEST_90")
# data_test = TestData("/home/ty/data/zsj/1/USLN-master/datasets/data", "SUID1_TEST_100")   # UIE_TEST_90   SUID1_TEST_100
test_iter = torch.utils.data.DataLoader(data_test, 1, drop_last=True,
                                        num_workers=1, pin_memory=True)



def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 2

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]



for id, (bag, name) in enumerate(test_iter):

    input = bag.to(device)
    with torch.no_grad():
        input_noisy, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy = pad_tensor(input)  # 这个是处理unet特征不对齐问题 调整图像大小

        st = time.time()
        gen_img = model(input_noisy)
        run_time = time.time() - st
        print("Running Time: {:.3f}s\n".format(run_time))
        print(torch.cuda.max_memory_allocated(device=None))
        output = pad_tensor_back(gen_img, pad_left_noisy, pad_right_noisy, pad_top_noisy, pad_bottom_noisy)




    save_image(output, "datasets/pred/%s.png" % (name[0]), nrow=1, normalize=False)



