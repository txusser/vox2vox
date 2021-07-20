from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import nibabel as nib

class Dataset_hf5(Dataset):
    
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = [x[0:-7] for x in glob.glob(self.datapath + '/*_in.nii')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_in = self.samples[idx] + '_in.nii'
        nii_out = self.samples[idx] + '_out.nii'
        image = nib.load(nii_in).get_fdata()
        mask = nib.load(nii_out).get_fdata()
        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)
        
        return {"A": image, "B": mask}
