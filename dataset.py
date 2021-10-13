from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import nibabel as nib
import torchio as tio

class torchio_dataset(object):
    
    def __init__(self, datapath, transforms = False):
        self.datapath = datapath
        self.samples = [x[0:-7] for x in glob.glob(self.datapath + '/*_in.nii')]


        if transforms:
            self.transforms = transforms
        else:
            self.transforms = tio.RescaleIntensity(out_min_max=(0, 1))

    def prepare_dataset(self):
        
        subjects_list=[]

        for idx in range(len(self.samples)):
            nii_in = self.samples[idx] + '_in.nii'
            nii_out = self.samples[idx] + '_out.nii'

            subject = tio.Subject(A=tio.ScalarImage(nii_in),B=tio.ScalarImage(nii_out))
            subjects_list.append(subject)
        
        subjects_dataset = tio.SubjectsDataset(subjects_list, transform=self.transforms)
        
        return subjects_dataset



        

