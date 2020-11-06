import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import torch
import cv2

class CustomDataset(Dataset):
  def __init__(self,csv_file,root_dir,folder_name,transform = None,resize = None):
    self.annotations = pd.read_csv(csv_file,index_col=0)
    self.folder_name = folder_name
    self.root_dir = root_dir
    self.transform = transform
    self.resize = resize

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    img_path = os.path.join(self.root_dir,self.folder_name,str(self.annotations.iloc[index,0])+".jpg")
    image = io.imread(img_path)
    #resize if needed
    if self.resize is not None:
        self.resize_image(image)
    y_label = torch.tensor(int(self.annotations.iloc[index,2]))

    if self.transform:
      image = self.transform(image)
    return(image,y_label)
  
  def resize_image(img, size=self.resize):
    """
    function takes input of any size and it creates a squared shape blank image of size image's height or width whichever is bigger.
    it then places the original image at the center of the blank image. 
    and then it resizes this square image into desired size so the shape of original image content gets preserved.
    """

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else 
                    cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)
