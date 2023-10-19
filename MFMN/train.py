import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

from NEW_ARCH import Encoder
from SegDataset import TrainLabeled
from loss import Combinedloss
from torchvision.utils import save_image
import torch.nn.functional as F
from math import log10
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from warmup_scheduler import GradualWarmupScheduler


########################################################
num_workers = 0 if sys.platform.startswith('win32') else 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#############################################################
torch.cuda.set_device(0)  # 指定GPU运行
if __name__ == "__main__":

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.initialized = False
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def initialize(self, val, weight):
            self.val = val
            self.avg = val
            self.sum = np.multiply(val, weight)
            self.count = weight
            self.initialized = True

        def update(self, val, weight=1):
            if not self.initialized:
                self.initialize(val, weight)
            else:
                self.add(val, weight)

        def add(self, val, weight):
            self.val = val
            self.sum = np.add(self.sum, np.multiply(val, weight))
            self.count = self.count + weight
            self.avg = self.sum / self.count

        @property
        def value(self):
            return self.val

        @property
        def average(self):
            return np.round(self.avg, 5)

    def to_psnr(J, gt):
        mse = F.mse_loss(J, gt, reduction='none')
        mse_split = torch.split(mse, 1, dim=0)
        mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
        intensity_max = 1.0
        psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
        return psnr_list

    def compute_psnr_ssim(recoverd, clean):
        assert recoverd.shape == clean.shape
        recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
        clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
        recoverd = recoverd.transpose(0, 2, 3, 1)
        clean = clean.transpose(0, 2, 3, 1)
        psnr = 0
        ssim = 0

        for i in range(recoverd.shape[0]):
            psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
            ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)

        return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

    Init_Epoch = 0
    Final_Epoch = 100
    batch_size = 1
    lr = 1e-2

    model = Encoder()
    save_model_epoch = 1

    model = model.to(device)



    data_train = TrainLabeled("/home/ty/data/zsj/1/USLN-master/datasets/data","train",256)
    data_test = TrainLabeled("/home/ty/data/zsj/1/USLN-master/datasets/data","val",256)

    myloss = Combinedloss().to(device)
    if True:
        batch_size = batch_size
        start_epoch = Init_Epoch
        end_epoch = Final_Epoch


        optimizer = optim.Adam(model.train().parameters(), lr=lr, weight_decay = 5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)


        for epo in range(start_epoch, end_epoch):
            train_loss = 0
            model.train()  # 启用batch normalization和drop out

            train_iter = torch.utils.data.DataLoader(data_train, batch_size, shuffle=True,
                                                     drop_last=True, num_workers=num_workers,pin_memory=True)
            test_iter = torch.utils.data.DataLoader(data_test, batch_size, drop_last=True,
                                                    num_workers=num_workers,pin_memory=True)

            for index, (bag, bag_msk) in enumerate(train_iter):

                bag = bag.to(device)
                bag_msk = bag_msk.to(device)
                optimizer.zero_grad()
                output = model(bag)

                loss = myloss(output, bag_msk)
                loss.backward()
                iter_loss = loss.item()

                train_loss += iter_loss
                optimizer.step()

                if np.mod(index, 15) == 0:
                    print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_iter), iter_loss))

            # 验证
            test_loss = 0
            model.eval()
            with torch.no_grad():
                psnr_val = []
                val_psnr = AverageMeter()
                val_ssim = AverageMeter()
                for index, (bag, bag_msk) in enumerate(test_iter):
                    bag = bag.to(device)
                    bag_msk = bag_msk.to(device)

                    optimizer.zero_grad()
                    output = model(bag)

                    temp_psnr, temp_ssim, N = compute_psnr_ssim(output, bag_msk)
                    val_psnr.update(temp_psnr, N)
                    val_ssim.update(temp_ssim, N)
                    psnr_val.extend(to_psnr(output, bag_msk))
                    print('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                        "Eval", epo, val_psnr.avg, val_ssim.avg))


                    ####################################  vision
                    if index % 2 == 0:
                        img_sample = torch.cat((bag, output, bag_msk), -1)
                        save_image(img_sample, "samples/%s.png" % (index), nrow=1, normalize=False)


            print('<---------------------------------------------------->')
            print('epoch: %f' % epo)
            print('epoch train loss = %f'
                  % (train_loss / len(train_iter)))

            lr_scheduler.step()
            # 每5个epoch存储一次模型
            if np.mod(epo, save_model_epoch) == 0:
                # 只存储模型参数
                torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % (
                    (epo + 1), (100*train_loss / len(train_iter)))
                           )
                print('saveing checkpoints/model_{}.pth'.format(epo))



