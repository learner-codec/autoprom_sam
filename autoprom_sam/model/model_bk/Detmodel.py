from .layers import *
import sys
sys.path.append("./SAM/")
from model.SAM.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import copy
import numpy as np
from .configs import CFG
from .losses import FocalLoss

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
    labels=[0,1,2,3,4]
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
    threshold = CFG.score_threshold  # NMS threshold, you can adjust this value as needed

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

    return [boxes],np.array(label_list)[idxs],scores_list



class DetNet(nn.Module):

    def __init__(self, num_classes,use_dense = True, block=None, layers=None):
        self.training = False
        self.inplanes = 64
        super(DetNet,self).__init__()
        
        self.feature_pyramid = FPN(use_dense=use_dense)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

        

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def forward(self, img_batch,all_features,annotations=None):


        ############################################
        features = self.feature_pyramid(all_features)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)
        #print(regression.shape)
        #print(anchors.shape)

        if annotations is not None:
            return self.focalLoss(classification.cpu(), regression.cpu(), anchors.cpu(), annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
        



class MaskHead(nn.Module):
    def __init__(self, image_embedding_size,image_size,prompt_embed_dim):
        super(MaskHead,self).__init__()
        self.prompt_encoder =PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        

        self.mask_decoder =MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def forward(self,batched_input,image_embeddings):
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            outputs.append(
                {
                    "masks": None,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        def process(logits):
            processed = []
            for i in range(logits.shape[0]):
                processed.append((logits[i]>0.0))
            return torch.stack(processed)
        binarized_logits = torch.stack([torch.sum(process(outputs[j]['low_res_logits']),axis=0)for j in range(len(outputs))], axis=1)
        return binarized_logits
         
    


class E2E(nn.Module):
    def __init__(self, num_classes, use_dense=True,attach_seg_head = True,train_bbox_decoder=True,train_seg_decoder=False,device="cuda"):
        super(E2E,self).__init__()
        self.features = []
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        self.training = False
        image_embedding_size = image_size // vit_patch_size
        self.device= device
        self.encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )
        self.detector = DetNet(num_classes,use_dense=CFG.use_dense)
        self.mask_head = MaskHead(image_embedding_size=image_embedding_size,image_size=image_size,prompt_embed_dim=prompt_embed_dim)

        self.train_bbox_decoder = train_bbox_decoder
        self.train_seg_decoder = train_seg_decoder
        self.attach_seg_head = attach_seg_head
        self.features = []
        #register a forward hook
        for i, block in enumerate(self.encoder.blocks):
            block.register_forward_hook(self.hook_fn)
        ###############################################

        #freeze the encoder. always ""
        for param in self.encoder.parameters():
            param.requires_grad = False

        #if not train decoder frezze the mask decoder
        if not self.train_seg_decoder and self.train_bbox_decoder:
            for name, param in self.mask_head.prompt_encoder.named_parameters():
                    param.requires_grad = False
            for name, param in self.mask_head.mask_decoder.named_parameters():
                    param.requires_grad = False
        elif not self.train_seg_decoder and not self.train_bbox_decoder:
            for name, param in self.mask_head.prompt_encoder.named_parameters():
                    param.requires_grad = False
            for name, param in self.mask_head.mask_decoder.named_parameters():
                    param.requires_grad = False
            for name, param in self.detector.named_parameters():
                    param.requires_grad = False
        else:
            for name, param in self.mask_head.prompt_encoder.named_parameters():
                    param.requires_grad = False

            for name, param in self.detector.named_parameters():
                    param.requires_grad = False

            




    def hook_fn(self,module, input, output):
        self.features.append(output)

    def forward(self,inputs):
        self.features = []
        annotations = None
        if self.training:
            img_batch, annotations = inputs
            annotations_de = self.process_proposals(copy.copy(annotations))
            batched_input = [{"image":img_batch[i],"boxes":torch.tensor(annotations_de[i], device=self.device),'original_size':img_batch[i].shape[1:]} for i in range(img_batch.shape[0])]
        else:
            img_batch = inputs
            batched_input = [{"image":img_batch[i],'original_size':img_batch[i].shape[1:]} for i in range(img_batch.shape[0])]
        
        img_batch = torch.stack([x["image"] for x in batched_input], dim=0)

        with torch.no_grad():
            out_ = self.encoder(img_batch)
        if self.train_bbox_decoder:
            self.detector.training = True
            #return the loss
            return self.__forward_bbox(img_batch,self.features,annotations)
        else:
            self.detector.training = False
            scores, classification, transformed_anchors = self.__forward_bbox(img_batch,self.features)
            pred_boxes, label_list,scores_list = filter(scores.cpu(), classification.cpu(), transformed_anchors.cpu())
            if self.attach_seg_head and not self.train_seg_decoder:
                proposals = self.process_proposals(pred_boxes)
                batched_input = [{"boxes":torch.tensor(proposals[i], device=self.device),'original_size':img_batch[i].shape[1:]} for i in range(img_batch.shape[0])]
                outputs = self.__forward_seg_head(batched_input,out_)
                return pred_boxes, label_list,scores_list,outputs
            if self.attach_seg_head and self.train_seg_decoder:
                outputs = self.__forward_seg_head(batched_input,out_)
                return outputs
            if not self.attach_seg_head:
                return pred_boxes, label_list,scores_list  

        

    def __forward_bbox(self,input,features,annotations=None):
        if annotations is not None:
            return self.detector(input,features,annotations)
        else: return self.detector(input,features)

    def __forward_seg_head(self,batched_input,image_embeddings):
         mask_logits = self.mask_head(batched_input,image_embeddings)
         return mask_logits
        
    def process_proposals(self,proposals):
        prop = []
        for prop_ in range(len(proposals)):
            prop.append(np.array(proposals[prop_])[:,:4])
        return prop
    
    def load_checkpoint_sam(self, path):
        if not hasattr(self, 'encoder'):
            raise ValueError("Encoder model not found. Please initialize self.encoder.")

        with open(path, "rb") as f:
            state_dict = torch.load(f)

        image_encoder_keys = [key for key in state_dict.keys() if 'image_encoder' in key]
        if not image_encoder_keys:
            raise ValueError("No image encoder keys found in the loaded checkpoint.")

        for key in image_encoder_keys:
            model_key = key.replace("image_encoder.", "")
            if model_key in self.encoder.state_dict().keys():
                self.encoder.state_dict()[model_key].copy_(state_dict[key])
            else:
                print(f"Warning: Key '{model_key}' not found in the encoder's state dictionary.")
        print("encoder key loaded")
        #load for the decoder
        #prompt encoder keys
        primpt_encoder_keys = [key for key in state_dict.keys() if 'prompt_encoder' in key]
        for key in primpt_encoder_keys:
            model_key = key.replace("prompt_encoder.", "")
            if model_key in self.mask_head.prompt_encoder.state_dict().keys():
                self.mask_head.prompt_encoder.state_dict()[model_key].copy_(state_dict[key])
            else:
                print(f"Warning: Key '{model_key}' not found in the encoder's state dictionary.")

        print(" prompt encoder key loaded")
        mask_decoder_keys = [key for key in state_dict.keys() if 'mask_decoder' in key]
        for key in mask_decoder_keys:
            model_key = key.replace("mask_decoder.", "")
            if model_key in self.mask_head.mask_decoder.state_dict().keys():
                self.mask_head.mask_decoder.state_dict()[model_key].copy_(state_dict[key])
            else:
                print(f"Warning: Key '{model_key}' not found in the encoder's state dictionary.")
        print("mask decoder key loaded")

        print("all state Loaded successfully")
        