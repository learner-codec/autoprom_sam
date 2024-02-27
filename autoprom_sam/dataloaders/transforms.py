import numpy as np
import random
from albumentations import (Compose,MedianBlur, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,BboxParams,
                            GridDistortion, RandomSizedBBoxSafeCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            RandomBrightness,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp,Resize)

from albumentations.pytorch.transforms import ToTensorV2
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def get_transforms_train(resize=[1024,1024]):
    transform_train = Compose([
        Resize(resize[0],resize[1]),
        RandomRotate90(p=1),
        HorizontalFlip(p=1),
        RandomSizedBBoxSafeCrop(width=resize[0], height=resize[1], erosion_rate=0.3,p=0.2),
        GaussNoise(var_limit=(0, 50.0), mean=0, p=0.25),
        GaussianBlur(blur_limit=(3, 7), p=0.2),
            # RandomBrightness (limit=0.2, always_apply=False, p=1),
        RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, brightness_by_max=True, p=0.2),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=0, p=0.2),
            # MedianBlur(blur_limit=(3, 7), p=1)
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]),
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ],bbox_params=BboxParams(format="yolo",label_fields=['category_id']),)
    return transform_train


def get_transforms_valid(resize=[1024,1024]):
    transform_valid = Compose([
        Resize(resize[0],resize[1]),
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]),
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ],bbox_params=BboxParams(format="yolo",label_fields=['category_id']),)
    return transform_valid
