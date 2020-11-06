import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import torch

class CustomDataset(Dataset):
  def __init__(self,csv_file,root_dir,folder_name,transform = None):
    self.annotations = pd.read_csv(csv_file,index_col=0)
    self.folder_name = folder_name
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    img_path = os.path.join(self.root_dir,self.folder_name,str(self.annotations.iloc[index,0])+".jpg")
    image = io.imread(img_path)
    y_label = torch.tensor(int(self.annotations.iloc[index,2]))

    if self.transform:
      image = self.transform(image)
    return(image,y_label)
