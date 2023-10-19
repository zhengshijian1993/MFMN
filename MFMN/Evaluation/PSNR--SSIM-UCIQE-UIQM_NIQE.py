'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:  图像运行后会有丢尺度问题，导致图像不对
python evaluate.py RESULT_PATH REFERENCE_PATH
'''
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.measure import compare_psnr, compare_ssim
import math
import sys
from skimage import io, color, filters
import os
import math
import numpy as np
from PIL import Image

## local libs
from imqual_utils import getSSIM, getPSNR
from niqe import niqe

def rmetrics(gtr_dir,gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """

    ssims, psnrs = [], []

    r_im = Image.open(gtr_dir).resize(im_res)
    g_im = Image.open(gen_dir).resize(im_res)
            # get ssim on RGB channels

    ssim = getSSIM(np.array(r_im), np.array(g_im))
    ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
    r_im = r_im.convert("L"); g_im = g_im.convert("L")
    psnr = getPSNR(np.array(r_im), np.array(g_im))
    psnrs.append(psnr)
    return np.array(psnrs), np.array(ssims)



def nmetrics(a, im_res=(256, 256)):

    a = np.array(Image.open(a).resize(im_res))
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top])-np.mean(sl[::top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int(al1 * len(rgl))
    T2 = np.int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]
            blockmin = np.float(np.min(block))
            blockmax = np.float(np.max(block))

            top = plipsub(blockmax,blockmin)
            bottom = plipsum(blockmax,blockmin)

            if math.isnan(top) or math.isnan(bottom) or bottom == 0.0 or top == 0.0:
                s += 0.0
            else:
                m = top / bottom  # 出现分母为0的情况
                if m == 0.:
                    s += 0
                else:
                    s += (m) * np.log(m)

    return plipmult(w,s)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main():

    result_path = "/home/ty/data/zsj/DC-net/checkpoints/Denoising/results/baseline/uie/"
    reference_path = "/home/ty/data/zsj/DC-net/data/test/GT/"    # SUID1_TEST_100    UIE_TEST_90  EUVP_TEST_270
    # result_dirs = os.listdir(result_path)
    reference_dirs = os.listdir(reference_path)

    sumpsnr, sumssim, sumuiqm, sumuciqe, sumniqe = 0.,0.,0.,0.,0.

    N=0
    for imgdir in reference_dirs:
        if is_image_file(imgdir):
            # reference image
            reference = os.path.join(reference_path, imgdir)

            #corrected image
            imgname = imgdir.split('.')[0]
            imgdir = imgname + ".jpg"
            corrected = os.path.join(result_path,imgdir)


            psnr,ssim = rmetrics(corrected,reference)         # 测试psnr，ssim

            uiqm,uciqe = nmetrics(corrected)                  # 测试uiqm，uciqe
            # NIQE = niqe(corrected)
            NIQE = 0
            sumpsnr += psnr
            sumssim += ssim
            sumuiqm += uiqm
            sumuciqe += uciqe
            sumniqe += NIQE
            N +=1

            with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
                f.write('{}: psnr={} ssim={} uiqm={} uciqe={} niqe={}\n'.format(imgname,psnr,ssim,uiqm,uciqe,NIQE))

            # with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
            #     f.write('{}: uiqm={} uciqe={} niqe={}\n'.format(imgname,uiqm,uciqe,NIQE))
    mpsnr = sumpsnr/N
    mssim = sumssim/N
    muiqm = sumuiqm/N
    muciqe = sumuciqe/N
    mniqe = sumniqe/N

    with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
        f.write('Average: psnr={} ssim={} uiqm={} uciqe={} niqe={}\n'.format(mpsnr, mssim, muiqm, muciqe, mniqe))
    # with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
    #     f.write('Average: uiqm={} uciqe={} niqe={}\n'.format( muiqm, muciqe, mniqe))
if __name__ == '__main__':
    main()