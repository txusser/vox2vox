import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from dataset import Dataset_hf5

from dice_loss import diceloss
import nibabel as nib

import torch.nn as nn
import torch.nn.functional as F
import torch

import h5py

def apply_model():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, default="model.pth", help="Pretrained model that will be used for inference")
    parser.add_argument("--in_path", type=str, default=False, help="Image to which the model will be applied")
    parser.add_argument("--out_path", type=str, default=False, help="Image to which the model will be applied")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False

    print("Cuda in use: %s" % str(cuda))

    generator = GeneratorUNet()
    generator.load_state_dict(torch.load(opt.model_pth))
    generator.eval()


    img = nib.load(opt.in_path)
    img_data = img.get_fdata()

    img_data = np.expand_dims(img_data, axis = 0)
    img_data = np.expand_dims(img_data, axis = 0)
    tensor_arr = torch.tensor(img_data)

    output = generator(tensor_arr).data.cpu().numpy()
    output = output[0,0,:,:,:]

    img_out = nib.Nifti1Image(output,img.affine,img.header)
    nib.save(img_out, opt.out_path)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    apply_model()
