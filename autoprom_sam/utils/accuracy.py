from tqdm import tqdm
import gc
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision
import sys
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

def nms_with_indices(boxes, scores, threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on the given list of bounding boxes.

    Parameters:
        boxes (List[List[float]]): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        scores (List[float]): A list of confidence scores corresponding to each bounding box.
        threshold (float): The overlap threshold to consider bounding boxes as duplicates.

    Returns:
        List[List[float]]: Filtered bounding boxes after NMS.
        List[float]: Filtered scores after NMS.
        List[int]: Selected indices after NMS.
    """
    if not boxes:
        return [], [], []

    # Sort the bounding boxes by their scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sorted_boxes = [boxes[i] for i in sorted_indices]

    filtered_boxes = []
    filtered_scores = []
    selected_indices = []

    while sorted_boxes:
        # Select the box with the highest score
        best_box = sorted_boxes[0]
        best_score = scores[sorted_indices[0]]
        best_idx = sorted_indices[0]

        filtered_boxes.append(best_box)
        filtered_scores.append(best_score)
        selected_indices.append(best_idx)

        # Compute IoU (Intersection over Union) with other boxes
        ious = [calculate_iou(best_box, box) for box in sorted_boxes[1:]]

        # Filter out boxes with IoU greater than the threshold
        remaining_indices = [i for i, iou in enumerate(ious) if iou < threshold]
        sorted_boxes = [sorted_boxes[i + 1] for i in remaining_indices]
        sorted_indices = [sorted_indices[i + 1] for i in remaining_indices]

    return filtered_boxes, filtered_scores, selected_indices


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (List[float]): The first bounding box in the format [x_min, y_min, x_max, y_max].
        box2 (List[float]): The second bounding box in the format [x_min, y_min, x_max, y_max].

    Returns:
        float: Intersection over Union (IoU) value.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou




def filter(scores, classification, transformed_anchors):
    labels = [0,1,2,3,4]
    """
    Filter bounding boxes using Non-Maximum Suppression (NMS).

    Parameters:
        scores (torch.Tensor): Tensor containing confidence scores for each bounding box.
        classification (torch.Tensor): Tensor containing predicted class labels for each bounding box.
        transformed_anchors (torch.Tensor): Tensor containing the transformed bounding boxes.

    Returns:
        List[torch.Tensor]: Filtered bounding boxes after NMS.
        List[str]: Filtered labels corresponding to each bounding box.
    """
    threshold = 0.3  # NMS threshold, you can adjust this value as needed

    # Convert tensors to numpy arrays
    scores_np = scores.cpu().numpy()
    classification_np = classification.cpu().numpy()
    transformed_anchors_np = transformed_anchors.cpu().numpy()

    # Filter bounding boxes based on confidence scores
    idxs = np.where(scores_np > threshold)
    boxes = [transformed_anchors_np[idx] for idx in idxs[0]]
    label_list = [labels[int(classification_np[idx])] for idx in idxs[0]]

    # Apply NMS
    boxes, scores_list,idxs= nms_with_indices(boxes, scores_np[idxs],threshold=0.2)

    return boxes,np.array(label_list)[idxs],scores_list


def valid_fn(data_loader, model):
    det = 0
    model.training = False
    metric = MeanAveragePrecision(class_metrics=True)
    tk0 = tqdm(data_loader, total=len(data_loader),file=sys.stdout)
    for iter_num,(id,img,mask,inst_mask,annot) in enumerate(tk0):
        with torch.no_grad():
            scores, classification, transformed_anchors = model(torch.stack(img).cuda().float())[0]

        pred_boxes, label_list,scores_list = filter(scores.cpu(), classification.cpu(), transformed_anchors.cpu())
        orig_boxes = np.array(annot)[0][:,:4].copy()
        orig_label = np.array(annot)[0][:,4].copy()
        try:
            targets=[
            {
                'boxes':torch.tensor(orig_boxes),
                'labels': torch.tensor(orig_label),
            }
            ]
            pred = [
                {
                "boxes":torch.tensor(pred_boxes),
                "scores":torch.tensor(scores_list),
                "labels":torch.tensor(label_list)
                }
            ]
            metric.update(pred,targets)
            det+=1
        except Exception as e:
            print(e)
        del img,annot,mask
        gc.collect()

        # tk0.set_postfix(loss=summary_loss.avg)
    if det>0:
        accuracy_metric = metric.compute()
    else:
        accuracy_metric = 0
    return 0,0,0,accuracy_metric





def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = groundtruth_box
    d_ymin, d_xmin, d_ymax, d_xmax = detection_box
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(original, prediction, num_classes, IOU_THRESHOLD=0.5):
    IOU_THRESHOLD = IOU_THRESHOLD
    CONFIDENCE_THRESHOLD = 0.3
    confusion_matrix = np.zeros(shape=(num_classes, num_classes))
    
    if original.shape[0] == 0:
        for i in range(prediction.shape[0]):
            predicted_class = int(prediction[i, 4])
            confusion_matrix[num_classes-1][predicted_class] += 1  # False Positive
        return confusion_matrix
    
    if prediction.shape[0] == 0:
        for i in range(original.shape[0]):
            groundtruth_class = int(original[i, 4])
            confusion_matrix[groundtruth_class][num_classes-1] += 1  # False Negative
        return confusion_matrix
    
    for i in range(original.shape[0]):
        groundtruth_box = original[i, :4]
        groundtruth_class = int(original[i, 4])
        detection_scores = prediction[:, 5]
        detection_classes = prediction[:, 4]
        detection_boxes = prediction[:, :4]
        
        matches = []
        
        for j in range(len(detection_boxes)):
            iou = compute_iou(groundtruth_box, detection_boxes[j])
            if iou > IOU_THRESHOLD:
                matches.append([i, j, iou])
        
        matches = np.array(matches)
        if matches.shape[0] > 0:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        
        if matches.shape[0] > 0 and (matches[:, 0] == i).any():
            matched_detection_class = int(detection_classes[int(matches[matches[:, 0] == i, 1][0])])
            confusion_matrix[groundtruth_class][matched_detection_class] += 1
    
    return confusion_matrix

def get_f1(confusion_matrix,num_classes):
    num_classes = num_classes
    precision = []
    recall = []
    f1_score = []

    for class_idx in range(num_classes):
        tp = confusion_matrix[class_idx, class_idx]
        fp = np.sum(confusion_matrix[:, class_idx]) - tp
        fn = np.sum(confusion_matrix[class_idx, :]) - tp

        precision_class = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall_class = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score_class = 2 * (precision_class * recall_class) / (precision_class + recall_class) if (precision_class + recall_class) != 0 else 0

        precision.append(precision_class)
        recall.append(recall_class)
        f1_score.append(f1_score_class)

    total_f1_score = np.mean(f1_score)

    # print("\nClass Names:", class_names)
    # print("\nPrecision:", precision)
    # print("Recall:", recall)
    # print("F1-Score:", f1_score)
    # print("\nTotal F1-Score:", total_f1_score)
    return total_f1_score


def calculate_f1_accuracy(targets,orig_label,pred_boxes,scores_list,label_list,num_classes):


    orig_b = torch.tensor(targets[:,:4].copy()).int()
    orig_l = torch.tensor(orig_label)

    pred_b = torch.tensor(pred_boxes).cpu().int()
    pred_s = torch.tensor(scores_list).cpu()
    pred_l = torch.tensor(label_list).cpu()
    orig = torch.cat([orig_b,orig_l.view(len(orig_l),1)],axis=1)
    pred_ = torch.cat([pred_b,pred_l.view(len(pred_l),1),pred_s.view(len(pred_s),1)],axis=1)

    num_classes = num_classes
    confusion_matrix = process_detections(orig.numpy(), pred_.numpy(), num_classes)
    #f1 = get_f1(confusion_matrix=confusion_matrix,num_classes=num_classes)
    return confusion_matrix