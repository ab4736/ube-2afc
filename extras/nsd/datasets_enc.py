import torch
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np
def to_torch(x):
    return torch.from_numpy(x)



class EncDataset(Dataset):
    def __init__(self, X, Y, sub, sub_num_voxels ,num_voxels_to_sample = int(5000), sample = True, preprocess = None):
        self.X = X
        self.Y = Y
        self.sub = sub
        self.sub_num_voxels = sub_num_voxels
        self.num_voxels_to_sample = num_voxels_to_sample
        self.sub_vox_start_ind = np.cumsum(sub_num_voxels)-sub_num_voxels
        self.sample = sample
        self.preprocess = preprocess
       
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if(self.preprocess is not None):
            x = self.preprocess(x)
        else:
            x = to_torch(x)
        y = self.Y[idx]
        sub = self.sub[idx]
        num_voxels = self.sub_num_voxels[sub]
        if(self.sample):
            vox_sel = np.random.randint(0,num_voxels,self.num_voxels_to_sample)
        else:
            vox_sel = np.arange(num_voxels)
        y = y[vox_sel]
        voxel_global_ind = vox_sel+self.sub_vox_start_ind[sub]
        return x , to_torch(y), to_torch(voxel_global_ind)


