from torch.utils.data import Dataset
import torch
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from .transforms import *


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean
class Load_Bbox_Kidney(Dataset):
    def __init__(self,mode="train",csv_file = None, test_ratio = None):
        self.df = pd.read_csv(csv_file)
        self.HEIGHT = 1024
        self.WIDTH = 1024
        if mode=='train':
            if test_ratio is not None:
                self.df = self.df[:test_ratio]
            else: self.df = self.df
            self.transforms = get_transforms_train()
        else:
            self.transforms = get_transforms_valid()
    def __len__(self):
        return len(self.df)
    
    def get_image(self, name):
        """Gets the image for a given row"""
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #image /= 255.0
        return img
    def get_mask(self,name):
        mask = cv2.imread(name.replace("images","masks").replace("_image","_mask"),0)
        return mask
    def rescale_bbox(self,boxes,width=1024,height=1024):
        bbox_list = []
        scaling_factor = 1024/256
        for bbox in boxes:
            x,y,w,h = bbox
            x = x*scaling_factor
            y = y*scaling_factor
            w = w*scaling_factor
            h = h*scaling_factor
            bbox_list.append((x,y,w,h))
        return bbox_list

    def xywh_to_xminyminxmaxymax(self,bbox):
        x_center, y_center, width, height = bbox
        x_min = (x_center - width/2) * self.WIDTH
        y_min = (y_center - height/2) * self.HEIGHT
        x_max = (x_center + width/2) * self.WIDTH
        y_max = (y_center + height/2) * self.HEIGHT
        return (x_min, y_min, x_max, y_max)
    def __getitem__(self, index):
        current_df = self.df.iloc[index]
        image = self.get_image(current_df["Image Path"])
        mask = self.get_mask(current_df["Image Path"])
        annotations = eval(current_df["bbox_coord"])
        n_boxes = len(annotations)
        
        target = {
            'boxes': torch.as_tensor(self.rescale_bbox(annotations), dtype=torch.float32),
            #'area': torch.as_tensor(area, dtype=torch.float32),
            
            # There is only one class
            'labels': torch.zeros((n_boxes,), dtype=torch.int64),
        }

        image_id = current_df["Image Name"]
        
        
        sample = {
            'bboxes': target['boxes'],
            'labels': target['labels']
        }
        # image = self.transforms(image=sample['image'])['image']
        #image = sample['image']

        #import pdb; pdb.set_trace()
        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        
        target['boxes'] = A.core.bbox_utils.normalize_bboxes(target['boxes'],rows=self.HEIGHT,cols=self.WIDTH)
        augmented = self.transforms(image=image.astype(np.uint8), 
                                        mask=mask.astype(np.int8),
                                        bboxes= target['boxes'],
                                         category_id=[0]*n_boxes )
        image = augmented["image"]
        mask = augmented["mask"]
        target['boxes'] = [self.xywh_to_xminyminxmaxymax(box) for box in augmented["bboxes"]]
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target["labels"] = torch.ones((len(augmented["bboxes"]),), dtype=torch.int64)
        annotations = np.zeros((len(target['boxes']), 5))
        annotations[:,:4] = target['boxes']
        annotations[:,4] = target["labels"]

        return image_id,image,mask,None,annotations