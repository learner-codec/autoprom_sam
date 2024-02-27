#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import os
from sklearn.model_selection import train_test_split
import random
from patchify import patchify
import tifffile as tiff
import gc
from PIL import Image, ImageDraw
import pandas as pd
BASE_WSI_PATH = '/home/humanoid/datasets/Hubmap_orig/'
# The tile sizes
size_x = 2048
size_y = 2048
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%
def file_list(folder):
    files_list = os.listdir(BASE_WSI_PATH+folder)
    files = []
    for file in files_list:
        if ".tiff" in file:
            files.append(file.split(".")[0])
    return files
#%%
def read_tiff(file_id,folder):
    '''
        This funciton will take a tiff file id with its folder location and read it. It should note that format of tiff
        files are not same across the folder. Some tiff images has more dims than other and sometimes the color channel
        is located differently. This need to checked manually before creating this funcion.
    '''
    print("Reading file ",file_id,end="")
    large_image_stack = tiff.imread(BASE_WSI_PATH+folder+'/'+file_id+'.tiff')#[0,0,:,:,:]
    print("Done")
    shape = large_image_stack.shape
    if(len(shape)>3):
        print("Image has more dims")
        large_image_stack = tiff.imread(BASE_WSI_PATH+folder+'/'+file_id+'.tiff')[0,0,:,:,:]
    else:
        print("Image has 3 dims")
    shape = large_image_stack.shape
    if shape[0]==3:
        print("Image is channel first, converting")
        large_image_stack = np.einsum('ijk->jki',large_image_stack)
    else:
        print("Image is channel last, no need to convert")
    shape = large_image_stack.shape
    print(shape)
    return large_image_stack, shape
# %%
import json
def read_mask(file_id,dir = "test"):    
    '''
        This function will take the id of the tiff image and serach for its mask in the training file
        folder. It first opens the json file, use geometry & coordinates fields to make polygons, draw
        the image using the polygons and returns it as the mask
    '''
    json_filename = BASE_WSI_PATH+f"test/"+file_id+'.json'
    read_file = open(json_filename, "r") 
    data = json.load(read_file)

    polys = []
    for index in range(data.__len__()):
        geom = np.array(data[index]['geometry']['coordinates'])
        polys.append(geom)
    #shape = (38160, 39000)

    Image.MAX_IMAGE_PIXELS = None
    mask = Image.new('L', (shape[1], shape[0]), 0)  # (w, h)
    for i in range(len(polys)):
        poly = polys[i]
        ImageDraw.Draw(mask).polygon(tuple(map(tuple, poly[0])), outline=i + 1, fill=i + 1) 
    polys = []
    mask = np.array(mask)
    return mask

# %%
test_files = file_list('test')
# %%
# 
import torch
import albumentations as A

# We can do many transformations
seq = A.Compose([
        A.HorizontalFlip(),
        A.Rotate(limit=(-45, 45)),
        # A.RandomCrop(128, 128),
        #A.ToTensor(),
    ])
seq_ = A.Compose([
        A.VerticalFlip(),
         A.ShiftScaleRotate(),
    ])
seq_s = A.Compose([
        #A.GaussianBlur(blur_limit=(3, 7),p=0.5),
        A.ColorJitter (brightness=0.05, contrast=0.01, saturation=0.01, hue=0.1, always_apply=False, p=0.5)
        #A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0)
        ])
seq_both = [seq,seq_]

def augment(image, mask,aug = random.choice(seq_both),aug2 = seq_s):
    #image = m#ia.quokka(size=(128, 128), extract="square")
    n = mask
    segmap = (n > 0).astype('uint8')
    #segmap = A.Segmentation(mask)
    transformed = aug(image = image,mask=n)
    aug1_img,aug1_mask = transformed['image'],transformed['mask']
    #aug 2 is not applied to the mask
    transformed_2 = aug2(image=image)
    aug2_img,aug2_mask = transformed_2["image"],n

    #fig, axs = plt.subplots(1,4,figsize=(10,10))
    #axs[0].imshow(cells[0])
    #axs[1].imshow(cells[1])
    #axs[2].imshow(cells[2])
    #axs[3].imshow(cells[3])
    return image,mask,aug1_img,aug1_mask,aug2_img,aug2_mask
# %%
print("rxtracting")
files = file_list('test') # to list all files in the train directory
c = 0
patch_length = size_x # tile size
step = int(size_x*0.75) # tile after every this much of pixels we have 25 % overlap
#create pandas dataframe
df = pd.DataFrame(columns=["Image_path","Mask_path","Image_orig_size"])
resize = 256 # whole image should be resized to this before saving it
items = 0
for file_id in files[:]:
    large_image_stack, shape = read_tiff(file_id,"test")
    # if needed do the resize as below, but it take lot of resources
    #large_image_stack = cv2.resize(large_image_stack,(shape[1]//3,shape[0]//3))
    print("New shape,",shape)
    # easy tiling of the large image
    patches_img = patchify(large_image_stack, (patch_length, patch_length, 3), step=step)
    # save memory
    large_image_stack = ""
    gc.collect()
    
    # do the same for the mask
    # mask = read_mask(file_id)
    # patches_mask = patchify(mask, (patch_length, patch_length), step=step)
    # mask = ""
    gc.collect()
    
    # calculate the percentage of the mask in the tiled image
    count = 0
    sums = []
    for row in range(patches_mask.shape[0]):
        for col in range(patches_mask.shape[1]):
            s = np.sum(patches_mask[row,col,:,:])
            count+=1
            sums.append([s, row, col, np.round(s/(patch_length*patch_length*255),2)])
    sums.sort()
    #if wanted filter images based on the coverage of the mask
    #ex: first 10% with zero masks + 40-60% mask coverage in the middle + more than 90% mask coverage
    #sums = sums[0:int(len(sums)*.10)]+sums[int(len(sums)*.40):int(len(sums)*.60)]+(sums[int(len(sums)*.90):])
    print(len(sums))
    c = 0
    for im in sums:
        c+=1
        row = im[1]
        col = im[2]

        # seperate corresponding image and its mask using sums list
        image = patches_img[row,col,0,:,:,:]
        image = cv2.resize(image, (resize, resize))
        # mask = patches_mask[row,col,:,:]
        # mask = cv2.resize(mask, (resize, resize))
        # mask = (mask>0).astype(int)
        #create uniqe file names
        filename_image = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_image.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"image.jpg"
        cv2.imwrite("/home/humanoid/datasets/Hubmap-kidney/test_patch_image/"+filename_image, image)
        #filename_mask = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_mask.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"mask.jpg"
        #cv2.imwrite("/home/humanoid/datasets/Hubmap-kidney/test_patch_mask/"+filename_mask, mask*255)
        df = df.append({"Image_path":filename_image,"Mask_path":filename_mask,"Image_orig_size":resize},ignore_index=True)
        """
        filename_image = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_image.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"image.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/image/"+filename_image, org_img)
        filename_mask = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_mask.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"mask.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/masks/"+filename_mask, org_msk*255)

        filename_image = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_image_aug0.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"image.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/image/"+filename_image, aug_img)
        filename_mask = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_mask_aug0.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"mask.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/masks/"+filename_mask, aug_mask*255)

        filename_image = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_image_aug1.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"image.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/image/"+filename_image, aug_img1)
        filename_mask = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_mask_aug1.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"mask.jpg"
        cv2.imwrite("/home/humanoid/internalHD/datasets/kidney_patch/train/masks/"+filename_mask, aug_mask1*255)
        """
        
    
        #use following if some randomness is required in saving images
#         num = random.randint(1,100)
#         #print("Num is {}".format(num))
#         if num>=50:
#             filename_image = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_image_aug.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"image.jpg"
#             cv2.imwrite("./images/"+filename_image, aug_img)
#             filename_mask = file_id+"_"+str("{0:0=3d}".format(row))+"_"+str("{0:0=3d}".format(col))+"_mask_aug.jpg"#file_id+"_"+str(str("{0:0=3d}".format(c)))+"mask.jpg"
#             cv2.imwrite("./masks/"+filename_mask, org_msk*255)
        print(".",end="")       
            #if im[3]>=.30: break
            #print(row, col, filename_image, filename_mask, "saved")
    df.to_csv("kidney_test_df.csv",index=False)
    print()
    print(items," ","Done",c," images")
    print()
    items+=1
    gc.collect()
    
# print("DONE=================")
# # %%
# # Inspeck few images ans its masks
# patch_path = "/home/humanoid/internalHD/datasets/kidney_patch/"
# ims = os.listdir(patch_path+'train/image')
# #mask = os.listdir(patch_path+'train/masks')
# random.shuffle(ims)
# msk = [i.replace("image","mask") for i in ims]
# ims = [f'{patch_path}train/image/'+x for x in ims]
# msk = [f'{patch_path}train/masks/'+x for x in msk]
# fig, axs = plt.subplots(1,2,figsize=(10,10))
# num = random.randint(1,len(ims))
# print(num)
# m = cv2.imread(ims[num])
# n = cv2.imread(msk[num])
# axs[0].imshow(m)
# axs[1].imshow(n)
# %%
