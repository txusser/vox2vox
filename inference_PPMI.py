import argparse
import os
import numpy as np
import pandas as pd
from  os.path import join,dirname,exists

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

def apply_model(model_pth,in_path,out_path):

    img_sample = nib.load(in_path)

    cuda = True if torch.cuda.is_available() else False
    Tensor =  torch.FloatTensor

    print("Cuda in use: %s" % str(cuda))

    generator = GeneratorUNet()
    generator.load_state_dict(torch.load(model_pth))
    generator.eval()

    transforms_ = transforms.Compose([transforms.ToTensor(),])

    image = load_nifti(in_path, transforms_)
    image =  Variable(image.unsqueeze_(0).type(Tensor))
    image =  Variable(image.unsqueeze_(0).type(Tensor))
    fake_B = generator(image)
    fake_B = fake_B.cpu().detach().numpy()[0,0,:,:,:]

    fake_B = np.swapaxes(fake_B,0,2)
    fake_B = np.swapaxes(fake_B,0,1)

    img_out = nib.Nifti1Image(fake_B,img_sample.affine,img_sample.header)
    nib.save(img_out, out_path)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

dat_csv = "C:/Users/jesus/OneDrive/FIDIS/Neuro/Quantidopa_v2/Datscan_data.csv"
source_dir = "D:/Work/DAT_SPECT/DaTSPECT/PPMI"
model_pth = "D:/Work/vox2vox/saved_models/datscan1/generator_195.pth"


orig_df = pd.read_csv(dat_csv)

for img_index, img_row in orig_df.iterrows():
    
    patient_id = str(img_row['Subject'])
    group_id = str(img_row['Group'])
    sex_id = str(img_row['Sex'])
    age_id = str(img_row['Age'])
    acq_date = str(img_row['Acq_Date'])
    scanner = str(img_row['Scanner_Model'])

    patient_path = join(source_dir,patient_id,'Reconstructed_DaTSCAN')
    if exists(patient_path):
        dir_list = os.listdir(patient_path)
        for i in dir_list:
            if i[0:10]==str(acq_date):
                
                normalization = 'None'
                nonorm_path = join(patient_path,i,'ndatscan.nii')
                dlnorm_path = join(patient_path,i,'dldatscan.nii')

                if exists(nonorm_path):
                    apply_model(model_pth,nonorm_path,dlnorm_path)
                    print('Created %s' % dlnorm_path)
