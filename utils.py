import math
import random
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from torch.autograd import Variable
import os
import glob


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, "*epoch*.pth"))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(
            Iclean[i, :, :, :, :], Img[i, :, :, :, :], data_range=data_range
        )
    return PSNR / Img.shape[0]


def normalize_NF(data):
    return data / 500


def de_normalize_NF(data):
    return data * 500
