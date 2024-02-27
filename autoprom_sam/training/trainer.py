import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import sys
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from autoprom_sam.configs.configs import CFG
from autoprom_sam.model.Detmodel import E2E
from autoprom_sam.dataloaders.data_loader_pannuke import Load_Bbox_nuclei
from autoprom_sam.utils.accuracy import valid_fn
from torch.utils.tensorboard import SummaryWriter
import gc
from autoprom_sam.training.run_utils import create_folders,load_weights_with_mismatched_keys,AverageMeter,collate_fn
CFG.configure('pannuke')
# Define command-line arguments
parser = argparse.ArgumentParser(description="Training script for detection model")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (CPU or GPU)")
args = parser.parse_args()


# Set random seed for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(args.seed)

def build_model():
    model = E2E(num_classes=CFG.num_classes,use_dense=CFG.use_dense,attach_seg_head=CFG.attach_seg_head,train_bbox_decoder=CFG.train_bbox_decoder,train_seg_decoder=CFG.train_seg_decoder,device=CFG.device)
    if CFG.resume:
        model = load_weights_with_mismatched_keys(model,CFG.most_recent_model)
    if CFG.load_sam_weights:
        model.load_checkpoint_sam(CFG.sam_checkpoint)
    model.training=True
    model.to(CFG.device)
    return model



def build_optimizer(model):
    optimizer_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=3e-4)
    if CFG.resume and CFG.recent_optimizer is not None:
        optimizer_state_dict = torch.load(CFG.recent_optimizer)
        optimizer.load_state_dict(optimizer_state_dict)
        print("optimizer state dict load success")
    return optimizer,optimizer_params


def build_schedular(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True,min_lr=3e-7)
    return scheduler

def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    loss_hist = []
    classification_losses = []
    regression_losses = []
    summary_loss = AverageMeter()

    with tqdm(data_loader, total=len(data_loader), file=sys.stdout) as tk0:
        for iter_num, (id, img, mask, inst_mask, annot) in enumerate(tk0):
            optimizer.zero_grad()

            classification_loss, regression_loss = model(
                [torch.stack(img).cuda().float(), np.array(annot)]
            )

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)

            optimizer.step()

            classification_losses.append(classification_loss.detach().item())
            regression_losses.append(regression_loss.detach().item())

            loss_hist.append(float(loss.item()))
            summary_loss.update(loss.item(), CFG.train_batch_size)

            del loss, img, annot, mask
            del classification_loss
            del regression_loss
            gc.collect()

            tk0.set_postfix(loss=summary_loss.avg)

    scheduler.step(np.mean(loss_hist))
    return (
        np.mean(classification_losses),
        np.mean(regression_losses),
        np.mean(loss_hist),
    )

def main():
    # Load data
    dataset_train = Load_Bbox_nuclei(csv_file=CFG.train_filename, mode="train", test_imgs=8)
    dataset_test = Load_Bbox_nuclei(csv_file=CFG.test_filename, mode="test", test_imgs=8)
    kf = KFold(n_splits=CFG.num_folds, shuffle=True)
    model = build_model()
    optimizer,opt_params = build_optimizer(model)
    scheduler = build_schedular(optimizer)
    print("total trainable parameters ",len(opt_params))
    model_ck,optim_ck,model_epoch_ck,model_fold_ck = create_folders(CFG.check_points_dir)
    logger = SummaryWriter("./runs/first_run_cnn_decoder_pannuke_dense_epoch_3plus")
    fold_accuracies = 0.00001
    epoch_accuracies = 0.00001
    counter = 0
    for epoch_num in range(CFG.EPOCHS):
        epoch_loss = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_train)):
            counter+=1
            # print(f"Fold {fold + 1}...")
            train_data = torch.utils.data.Subset(dataset_train, train_idx)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=CFG.train_batch_size,num_workers=CFG.num_workers,collate_fn=collate_fn, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=CFG.val_batch_size,num_workers = CFG.num_workers,collate_fn=collate_fn)
            class_loss, reg_loss , total_loss = train_fn(train_loader, model, optimizer, device=CFG.device,scheduler=scheduler,epoch=epoch_num)
            if CFG.record:
                torch.save(model.state_dict(), os.path.join(f'{model_ck}',f'detector_last_checkpoint.bin'))
                torch.save(optimizer.state_dict(), os.path.join(f'{optim_ck}',f'optimizer_last_checkpoint.bin'))
            epoch_loss.append(np.mean(total_loss))
            logger.add_scalar("class_loss",class_loss,counter)
            logger.add_scalar("regression_loss",reg_loss,counter)
            logger.add_scalar("fold_loss",np.mean(total_loss),counter)
            print('Epoch: {} |Fold: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} |epoch loss: {:1.5f}'.format(
                        epoch_num, fold+1, float(class_loss), float(reg_loss),np.mean(total_loss)))
            _,_,_,accuracy_metric = valid_fn(val_loader,model)
            if accuracy_metric!=0:
                torch.save(model.state_dict(), os.path.join(f'{model_ck}',f'detector_best_fold_checkpoint.bin'))
                print("mAp: ",accuracy_metric["map"].item(),"mAp_per_class: ",accuracy_metric["map_per_class"].tolist() )
            
                if CFG.record:
                    if fold_accuracies<accuracy_metric["map"].item():
                        fold_accuracies =accuracy_metric["map"].item()
                        best_loss = np.mean(total_loss)
                        print("best loss found==> ",np.mean(total_loss))
                        torch.save(model.state_dict(), os.path.join(f'{model_fold_ck}',f'pytorch_model_best_fold_retine_sam_{str(epoch_num+1)}_{str(fold+1)}.bin'))

        _,_,_,accuracy_metric = valid_fn(val_loader,model)
        if accuracy_metric!=0:
            print("mAp: ",accuracy_metric["map"].item(),"mAp_per_class: ",accuracy_metric["map_per_class"].tolist() )
            logger.add_scalar("accuracy",accuracy_metric["map"].item(),epoch_num)
            if CFG.record and epoch_accuracies<accuracy_metric["map"].item():
                epoch_accuracies = accuracy_metric["map"].item()
                torch.save(model.state_dict(), os.path.join(f'{model_ck}',f'detector_best_epoch_checkpoint.bin'))

        logger.add_scalar("epoch loss",np.mean(epoch_loss),epoch_num)
        logger.flush()

if __name__ == "__main__":
    main()

    
