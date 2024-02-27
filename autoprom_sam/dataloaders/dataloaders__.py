import sys
sys.path.append("../../")
from .data_loader_kidney import Load_Bbox_Kidney
from ..configs.configs import CFG
def get_dataloader_train():
    return Load_Bbox_Kidney(csv_file=CFG.train_filename,mode="train",test_ratio=16)