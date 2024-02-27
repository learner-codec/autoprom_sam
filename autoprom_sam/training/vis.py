import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import time

denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
import os

use_gpu = True


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
def vis(model,val_loader,fold,epoch):
    model.training = False
    for iter_num,(id,img,mask,inst_mask,annot) in enumerate(val_loader):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                preds = model(img[0].unsqueeze(0).cuda().float())
            scores, classification, transformed_anchors = preds[0]
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)
            img = denormalize(img[0]).permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)


            #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                # label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                # draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                # print(label_name)
            # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_save_path = os.path.join("/autoprom_sam/detected_image", f"{fold}_{epoch}_detection.jpg")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_save_path, img_bgr)
        break
