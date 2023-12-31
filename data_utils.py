import time
import os
import random
import numpy as np
import torch
import torch.utils.data

from torchvision import transforms

from utils import read_image

WBC_mean = np.array([0.7049, 0.5392, 0.5885])
WBC_std = np.array([0.1626, 0.1902, 0.0974])

class WBCdataset(torch.utils.data.Dataset):
    """
    Load WBC dataset
    """
    def __init__(self, path_to_folder, label_dict, transform):
        self.image_paths = []
        self.labels = []
        self.mask_paths = {}
        self.transform = transform
        
        for label_name in label_dict.keys():
            img_folder_path = os.path.join(path_to_folder, 'data/' + label_name)
            msk_folder_path = os.path.join(path_to_folder, 'mask/' + label_name)
            for f in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, f)
                msk_path = os.path.join(msk_folder_path, f)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label_dict[label_name])
                    if os.path.exists(msk_path):
                        self.mask_paths[len(self.image_paths)-1] = msk_path
            
        
    def __getitem__(self, index):
        img = read_image(self.image_paths[index])
        label = self.labels[index]
        #mask = read_image(self.mask_paths[index]) if index in self.mask_paths.keys() else None
        
        return self.transform(img), label
    
    def __len__(self):
        return len(self.labels)

class WBCdataset_Mask(torch.utils.data.Dataset):
    """
    Load WBC dataset
    """
    def __init__(self, path_to_folder, label_dict, transform, use_mask=False, is_train=False):
        self.use_mask = use_mask
        self.image_paths = []
        self.labels = []
        self.mask_paths = {}
        self.transform = transform
        
        self.is_train = is_train
        self.flip_transform = transforms.RandomHorizontalFlip()
        
        self.norm_transform = transforms.Normalize(WBC_mean, WBC_std, inplace=True)
        
        self.dummy_mask = torch.ones(3, 224, 224)
        
        for label_name in label_dict.keys():
            img_folder_path = os.path.join(path_to_folder, 'data/' + label_name)
            msk_folder_path = os.path.join(path_to_folder, 'mask/' + label_name)
            for f in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, f)
                msk_path = os.path.join(msk_folder_path, f)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label_dict[label_name])
                    if os.path.exists(msk_path):
                        self.mask_paths[len(self.image_paths)-1] = msk_path
            
        
    def __getitem__(self, index):
        img = read_image(self.image_paths[index])
        label = self.labels[index]
        mask = read_image(self.mask_paths[index]) if index in self.mask_paths.keys() else None
        
        if mask is None or self.use_mask is False:
            if self.is_train:
                return self.flip_transform(self.norm_transform(self.transform(img))), label, self.dummy_mask
            else:
                return self.norm_transform(self.transform(img)), label, self.dummy_mask
        else:
            img, mask = self.norm_transform(self.transform(img)), self.transform(mask)
            img_mask = torch.cat((img.unsqueeze(0), mask.expand(3, -1, -1).unsqueeze(0)),0)
            img_mask = self.flip_transform(img_mask)
            
            img, mask = img_mask[0], img_mask[1]
            p = torch.rand(1).item()
            if p > 0.5:
                mask = self.dummy_mask
            return img, label, mask
    
    def __len__(self):
        return len(self.labels)
    
class pRCCdataset(torch.utils.data.Dataset):
    """
    Load pRCC dataset
    """
    def __init__(self, path_to_folder, transform):
        self.image_paths = []
        self.transform = transform
        
        for f in os.listdir(path_to_folder):
            img_path = os.path.join(path_to_folder, f)
            if os.path.isfile(img_path):
                self.image_paths.append(img_path)
                
    def __getitem__(self, index):
        img = read_image(self.image_paths[index])
        return self.transform(img), []
    
    def __len__(self):
        return len(self.image_paths)
    
class CAM16dataset(torch.utils.data.Dataset):
    """
    Load CAM16 dataset
    """
    def __init__(self, path_to_folder, transform):
        self.image_paths = []
        self.transform = transform
        
    def __getitem__(self, index):
        img = read_image(self.image_paths[index])
        return self.transform(img), []
    
    def __len__(self):
        return len(self.image_paths)