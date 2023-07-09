import os

import cv2
import glob
import torch
from torch.utils.data import Dataset

class Train_dataset(Dataset):
    def __init__(self, root_dir):
        self.images_path = glob.glob(root_dir + '/*.jpg')
    
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path=self.images_path[index]
        image_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        label = 0
        label_one_hot = torch.tensor([1, 0, 0])
        if 'hopper' in img_path:
            label = 1
            label_one_hot=torch.tensor([0, 0, 1])
        elif 'tank' in img_path:
            label = 2
            label_one_hot=torch.tensor([0, 1, 0])
        ##  Доделвть лейблы
        ##  Сделать предобработку картинок

        return {
            'image': image,
            'label': label,
        }
    
dataset = Train_dataset('./')
dataset[0]['image']
