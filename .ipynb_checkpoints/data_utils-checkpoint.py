import time
import os
import random
import numpy as np
import torch
import torch.utils.data

class WBCdataset(torch.utils.data.Dataset):
    """
    Load WBC dataset
    """
    def __init__(self, path_to_folder, label_dict):
        self.image_paths = []
        self.labels = []
        self.mask_paths = {}
        
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
    
    def __len__(self):