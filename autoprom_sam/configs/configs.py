class CFG_KIDNEY:
    train_filename = "../../dataset/filtered_dataset.csv"
    test_filename = None
    most_recent_model = "./checkpoints_dense_unit_aug_ep_100/model_fold_checkpoints/pytorch_model_state_recent_l.bin"
    recent_optimizer = None
    sam_checkpoint = "../../sam_weights/sam_vit_b_01ec64.pth"
    check_points_dir = "./dry_run_kidney_ep_100/"
    log_dir = "./runs/dry_run_kidney_ep_100"
    resume = False
    use_transposed = True
    record =True
    load_sam_weights = True
    device = 'cuda'
    num_workers = 8
    train_batch_size = 16
    val_batch_size = 1
    num_folds = 3
    eval_period = 10
    #model speciic
    num_classes=2
    use_dense=True
    attach_seg_head=False
    train_bbox_decoder=True
    train_seg_decoder=False
    dense_units = [2,4,6,6]
    fatter = False ## create two more layers in classification network
    #dense_units = [6,6,10,10]
    #configurations
    EPOCHS = 100
    seed = 42
    #############################################
    #for inference
    bbox_iou = 0.5
    score_threshold = 0.98
    labels=[0,1]

class CFG_PANNUKE:
    ####
    #if you want to continue training
    train_filename = "/home/humanoid/internalHD/autoprom_sam/autoprom_sam/datasets/train_df.csv"
    test_filename = "/home/humanoid/internalHD/autoprom_sam/autoprom_sam/datasets/test_df.csv"
    most_recent_model = "./checkpoints_dense_unit_aug_ep_100/model_fold_checkpoints/pytorch_model_state_recent_l.bin"
    ### End
    recent_optimizer = None
    sam_checkpoint = "/home/humanoid/internalHD/autoprom_sam/sam_weights/sam_vit_b_01ec64.pth"
    check_points_dir = "./checkpoints/"
    log_dir = "./runs/logs"
    resume = False
    use_transposed = True
    record =True
    load_sam_weights = True
    device = 'cuda'
    num_workers = 8
    train_batch_size = 2
    val_batch_size = 1
    num_folds = 3
    eval_period = 10
    #model speciic
    num_classes=5
    use_dense=True
    attach_seg_head=False
    train_bbox_decoder=True
    train_seg_decoder=False
    dense_units = [2,4,6,6]
    fatter = False
    EPOCHS = 100
    seed = 42
    #############################################
    #for inference
    bbox_iou = 0.5
    score_threshold = 0.3
    labels=[0,1,2,3,4]

class CFG_tmp:
    @classmethod
    def configure(cls,dataset):
        if dataset == 'kidney':
            k_ = [keys for keys in CFG_KIDNEY.__dict__ if not "__" in keys]
            for k in k_:
                setattr(cls,k,CFG_KIDNEY.__dict__[k])
        elif dataset == "pannuke":
            k_ = [keys for keys in CFG_PANNUKE.__dict__ if not "__" in keys]
            for k in k_:
                setattr(cls,k,CFG_PANNUKE.__dict__[k])
        else:
            print("******* dataset configuration not found *******************************")
        

CFG = CFG_tmp
