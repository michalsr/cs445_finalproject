from PIL import Image
import os
import numpy as np
import json
import random
from torchvision.transforms import ToPILImage
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.datasets
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image
import shutil
from jpeg2dct.numpy import load, loads
from transforms import *
from create_dataset import *
from tqdm import tqdm
def calculate_channel_mean(channel):
    cuda = torch.device('cuda')
    means = torch.zeros(64,device=cuda)
    stds = torch.zeros(64,device=cuda)
    for i in range(64):

        #y = torch.flatten(channel[:,:,i])

        means[i] = torch.mean(torch.flatten(channel[i,:,:]))
        stds[i] = torch.std(torch.flatten(channel[i,:,:]))

    return means,stds


#as done at the bottom of https://github.com/calmevtime/DCTNet/blob/master/classification/datasets/dataset_imagenet_dct.py
def calculate_mean():
    dataset = ImagenetteTrain('train')
    cuda = torch.device('cuda')
    dct_y_mean = np.zeros((len(dataset),64))
    dct_y_std = np.zeros((len(dataset),64))
    dct_cb_mean = np.zeros((len(dataset),64))
    dct_cb_std = np.zeros((len(dataset),64))
    dct_cr_mean = np.zeros((len(dataset),64))
    dct_cr_std = np.zeros((len(dataset),64))
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    for batch_idx, (dct_y,dct_cb,dct_cr,targets) in tqdm(enumerate(data_loader)):
        print(batch_idx)
        dct_y, dct_cb, dct_cr = dct_y.numpy(),dct_cb.numpy(),dct_cr.numpy()
        print(dct_y.shape)
        print(dct_cb.shape)
        print(dct_cr.shape)
        #dct_y,dct_cb,dct_cr = dct_y.to(cuda),dct_cb.to(cuda),dct_cr.to(cuda)
        #print(dct_y.shape)
        dct_y_mean[batch_idx], dct_y_std[batch_idx] = np.mean(dct_y[0],axis=(0,1)),np.std(dct_y[0],axis=(0,1))
        dct_cb_mean[batch_idx], dct_cb_std[batch_idx] = np.mean(dct_cb[0],axis=(0,1)), np.std(dct_cb[0],axis=(0,1))
        dct_cr_mean[batch_idx], dct_cr_std[batch_idx] = np.mean(dct_cr[0],axis=(0,1)),np.std(dct_cr[0],axis=(0,1))
        # y_std = torch.std(dct_y,axis=(0,1))
        # dct_y_std[batch_idx] = y_std
        # cb_mean = torch.mean(dct_cb,axis=(0,1))
        # dct_cb_mean[batch_idx] = cb_mean
        # cb_std = torch.std(dct_cb,axis=(0,1))
        # dct_cb_std[batch_idx] = cb_std
        # cr_mean = torch.mean(dct_cr,axis=(0,1))
        # dct_cr_mean[batch_idx] = cr_mean
        # cr_std = torch.std(dct_cr,axis=(0,1))
        # dct_cr_std[batch_idx] = cr_std


    mean = np.mean(dct_y_mean,axis=0),np.mean(dct_cb_mean,axis=0),np.mean(dct_cr_mean,axis=0)
    std = np.mean(dct_y_std,axis=0),np.mean(dct_cb_mean,axis=0),np.mean(dct_cr_std,axis=0)
    np.save('/home/michal5/cs445/avgs_imagenette_full.npy',mean)
    np.save('/home/michal5/cs445/stds_imagenette_full.npy',std)

def test_shape():
    mean = np.load('/home/michal5/cs445/avgs.npy')
    print(mean.shape)
    stds = np.load('/home/michal5/cs445/stds.npy')
    print(stds.shape)

calculate_mean()

