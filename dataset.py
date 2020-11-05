#dataset.py

import torch
import numpy as np
import cv2

from PIL import Image

class ClassificationDataset :
    def __init__(self,
                 image_paths,
                 targets,
                 resize = None , 
                 augmentations = None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmantations = augmentations
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.image_paths)
    
    def __getitem__(self,item):
        """
        For a given "item" index, return everything we need 
        to train a given model
        """
        # Read the image
        image = cv2.imread(self.image_paths[item])
        
        # grab the targets
        targets = self.targets[item]
        
        #resize if needed
        if self.resize is not None:
            image = cv2.resize(image,
                               (self.resize[1],
                                self.resize[0]),interpolation = cv2.INTER_AREA)
            
        # pytorch expects CHW instead of HWC
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        
        return{
            "image" : torch.tensor(image,dtype = torch.float),
            "targets" : torch.tensor(targets,dtype = torch.long),
            }
        
