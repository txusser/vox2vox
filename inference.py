import argparse
import os
import numpy as np
from  os.path import join,dirname

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *

from dice_loss import diceloss
import nibabel as nib

import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py

def load_nifti(img, transforms=False):
    image = nib.load(img).get_fdata()
    image = image/np.max(image)

    if transforms:
        image = transforms(image)
        
    return image

def apply_model():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, default="model.pth", help="Pretrained model that will be used for inference")
    parser.add_argument("--in_path", type=str, default=False, help="Image to which the model will be applied")
    parser.add_argument("--out_path", type=str, default=False, help="Image to which the model will be applied")
    opt = parser.parse_args()
    print(opt)

    img_sample = nib.load(opt.in_path)

    cuda = True if torch.cuda.is_available() else False
    Tensor =  torch.FloatTensor

    print("Cuda in use: %s" % str(cuda))

    generator = GeneratorUNet()
    generator.load_state_dict(torch.load(opt.model_pth))
    generator.eval()

    transforms_ = transforms.Compose([transforms.ToTensor(),])


    image = load_nifti(opt.in_path, transforms_)
    image =  Variable(image.unsqueeze_(0).type(Tensor))
    image =  Variable(image.unsqueeze_(0).type(Tensor))
    fake_B = generator(image)
    fake_B = fake_B.cpu().detach().numpy()[0,0,:,:,:]

    fake_B = np.swapaxes(fake_B,0,2)
    fake_B = np.swapaxes(fake_B,0,1)


    img_out = nib.Nifti1Image(fake_B,img_sample.affine,img_sample.header)
    nib.save(img_out, opt.out_path)

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    apply_model()
