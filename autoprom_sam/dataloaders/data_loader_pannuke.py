import sys
from torch.utils.data import Dataset
import torch
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import random
import re
from .transforms import *


def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean
class Load_Bbox_nuclei(Dataset):
    def __init__(self,mode="train",csv_file = None,test_imgs = None):
        self.df = pd.read_csv(csv_file)
        if mode=='train':
            if test_imgs is not None:
                self.df = self.df[:test_imgs]
            else:
                self.df = self.df
            self.transforms = get_transforms_valid
        elif mode =='test':
            if test_imgs is not None:
                self.df = self.df[:test_imgs]
            self.transforms = get_transforms_valid
        else:
            if test_imgs is not None:
                self.df = self.df[:test_imgs]
            else:
                self.df = self.df
            self.transforms = get_transforms_valid
        self.mode = mode
        self.HEIGHT = 256
        self.WIDTH = 256
        #for resizing purposes
        self.HEIGHT_ = 1024
        self.WIDTH_ = 1024
        
    def __len__(self):
        return len(self.df)
    
    def get_image(self, name):
        pattern = r"pannuke_processed//[^/]+/images/"
        name = re.sub(pattern, f"pannuke_split/{self.mode}/images/", name)
        img = cv2.imread(name)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    def get_inst_mask(self,name):
        pattern = r"pannuke_processed//[^/]+/inst_masks/"
        name = re.sub(pattern, f"pannuke_split/{self.mode}/inst_masks/", name)
        mask = cv2.imread(name,0)
        return mask
    def get_sem_mask(self,name):
        pattern = r"pannuke_processed//[^/]+/sem_masks/"
        name = re.sub(pattern, f"pannuke_split/{self.mode}/sem_masks/", name)
        mask = cv2.imread(name,0)
        return mask
    def xminyminxmaxymax_to_cxcywh(self, boxes, image_width, image_height):
        box_list = []
        for bbox in boxes:
            bbox_start_w, bbox_start_h, bbox_end_w, bbox_end_h = bbox
            bbox_start_w_norm = bbox_start_w / image_width
            bbox_start_h_norm = bbox_start_h / image_height
            bbox_end_w_norm = bbox_end_w / image_width
            bbox_end_h_norm = bbox_end_h / image_height
            
            bbox_width_norm = bbox_end_w_norm - bbox_start_w_norm
            bbox_height_norm = bbox_end_h_norm - bbox_start_h_norm
            
            cx_norm = max(bbox_start_w_norm + (bbox_width_norm / 2),0.0001)
            cy_norm = max(bbox_start_h_norm + (bbox_height_norm / 2),0.0001)
            
            w_norm = max(bbox_width_norm,0.0001)
            h_norm = max(bbox_height_norm,0.0001)
            
            box_list.append((cx_norm, cy_norm, w_norm, h_norm))

        return box_list
    
    def denormalize_bbox_coordinates_2(self,bbox):
        x_center, y_center, width, height = bbox
        x_min = (x_center - width / 2)  * self.WIDTH_
        y_min = (y_center - height / 2) * self.HEIGHT_
        x_max = (x_center + width / 2) * self.WIDTH_
        y_max = (y_center + height / 2) * self.HEIGHT_

        return [x_min, y_min, x_max, y_max]
    


    def __getitem__(self, index):
        image = self.get_image(self.df.iloc[index]["Image_File"])
        inst_mask = self.get_inst_mask(self.df.iloc[index]["Inst_Mask_File"])
        sem_mask = self.get_sem_mask(self.df.iloc[index]["Sem_Mask_File"])
        boxes = eval(self.df.iloc[index]["BBox"])
        classes = eval(self.df.iloc[index]["Label"])
        target = {
            'boxes': torch.as_tensor(self.xminyminxmaxymax_to_cxcywh(boxes,self.WIDTH,self.HEIGHT),dtype=torch.float32),
            'labels': torch.tensor(([item for item in classes]), dtype=torch.int64),
        }
        
        
        sample = {
            'bboxes': target['boxes'],
            'labels': target['labels']
        }
        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).contiguous()
        #target["boxes"]=[tuple(torch.tensor(element.detach()) for element in sublist) for sublist in target["boxes"]]
        augmented = self.transforms(resize=(self.WIDTH_,self.HEIGHT_))(image=image.astype(np.uint8), 
                                    mask=sem_mask.astype(np.uint8),
                                    bboxes= target['boxes'],
                                    category_id=target["labels"] )
        image = augmented["image"]
        sem_mask = augmented["mask"]
        target['boxes'] = [self.denormalize_bbox_coordinates_2(box) for box in augmented["bboxes"]] #x_min, y_min, x_max, y_max
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target["labels"] = torch.tensor(([item.item() for item in augmented["category_id"]]), dtype=torch.int64)
        annotations = np.zeros((len(target['boxes']), 5))
        annotations[:,:4] = target['boxes']
        annotations[:,4] = target["labels"]-1
        return self.df.iloc[index]["Image_File"],image,sem_mask,inst_mask,annotations