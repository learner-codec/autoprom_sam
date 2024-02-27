import torch
import os
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_weights_with_mismatched_keys(model, weights_file):
    state_dict = torch.load(weights_file)

    model_dict = model.state_dict()
    mismatched_keys = []

    # Check for key mismatches
    for key in state_dict:
        if key in model_dict:
            # Load weights for matching keys
            model_dict[key] = state_dict[key]
        else:
            # Keep track of mismatched keys
            mismatched_keys.append(key)

    # Load the updated state_dict to the model
    model.load_state_dict(model_dict)

    # Print mismatched keys, if any
    if len(mismatched_keys) > 0:
        print("***********************Mismatched keys found:")
        for key in mismatched_keys:
            print(key)
    else:
        print("***********all key matched ***************")

    return model

def create_folders(root):
    folders = [
        "model_checkpoints",
        "optimizers_checkpoints",
        "model_epoch_checkpoints",
        "model_fold_checkpoints",
    ]
    for folder in folders:
        os.makedirs(os.path.join(root, folder), exist_ok=True)
    return [os.path.join(root, folder) for folder in folders]

def collate_fn(batch):
    return tuple(zip(*batch))

