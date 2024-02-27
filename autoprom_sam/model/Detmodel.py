from .layers import *
import sys
from .SAM.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import copy
import numpy as np
from ..configs.configs import CFG
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
    labels=CFG.labels
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

    def __init__(self, num_classes,use_dense = True, block=None, layers=None,device='cpu'):
        self.training = False
        self.inplanes = 64
        super(DetNet,self).__init__()
        self.device = device
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
            results = []
            for batch in range(classification.shape[0]):
                finalResult = [[], [], []]

                finalScores = torch.Tensor([])
                finalAnchorBoxesIndexes = torch.Tensor([]).long()
                finalAnchorBoxesCoordinates = torch.Tensor([])

                if self.device!='cpu':
                    finalScores = finalScores.cuda()
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
                else:
                    finalScores = finalScores
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates
                for i in range(classification.shape[2]):
                    scores = torch.squeeze(classification[batch, :, i])
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                        # no boxes to NMS, just continue
                        #result.append([torch.Tensor([]).float()]*3)
                        continue
                    scores = scores[scores_over_thresh]
                    anchorBoxes = torch.squeeze(transformed_anchors[batch])
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                    finalResult[0].extend(scores[anchors_nms_idx])
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                    if self.device!='cpu':
                        finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()


                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
                results.append([finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates])

            return results



class MaskHead(nn.Module):
    def __init__(self, image_embedding_size, image_size, prompt_embed_dim, return_multi_label=True, device='cpu'):
        super(MaskHead, self).__init__()
        self.return_multi_label = return_multi_label
        self.device = device
        
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
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

    def forward(self, batched_input, image_embeddings):
        outputs = self._process_inputs(batched_input, image_embeddings)
        return self._process_outputs(outputs)

    def _process_inputs(self, batched_input, image_embeddings):
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            points = self._get_points(image_record)
            sparse_embeddings, dense_embeddings = self._get_embeddings(image_record, points)
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            
            low_res_masks = self._filter_masks(low_res_masks, iou_predictions)
            outputs.append(
                {
                    "masks": None,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "labels": image_record.get("labels", None)
                }
            )
        return outputs

    def _get_points(self, image_record):
        if "point_coords" in image_record:
            return image_record["point_coords"], image_record["point_labels"]
        return None

    def _get_embeddings(self, image_record, points):
        with torch.no_grad():
            return self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

    def _filter_masks(self, low_res_masks, iou_predictions):
        highest_conf_indices = np.argmax(iou_predictions.cpu(), axis=1)
        low_res_masks = low_res_masks.cpu()
        filtered_masks = np.empty((len(low_res_masks), 1, 256, 256))
        
        for mask_i in range(len(low_res_masks)):
            filtered_masks[mask_i, 0] = low_res_masks[mask_i, highest_conf_indices[mask_i]]
        
        return torch.tensor(filtered_masks)

    def _process_outputs(self, outputs):
        if not self.return_multi_label:
            return torch.stack([self._process_logits(output['low_res_logits']) for output in outputs], axis=0)
        else:
            return torch.stack([self._process_multi_l(output['low_res_logits'], output["labels"]) for output in outputs], axis=1)

    def _process_logits(self, logits):
        return torch.stack([(logit > 0.0) for logit in logits])

    def _process_multi_l(self, logits, labels):
        accumulated_mask = np.zeros((1, 256, 256), dtype=np.int32)
        accumulated_instance = np.zeros((1, 256, 256), dtype=np.int32)
        
        for i in range(logits.shape[0]):
            curr_mask = (logits[i] > 0.0).cpu().numpy()
            accumulated_mask = np.where(curr_mask > 0, labels[i] + 1, accumulated_mask)
            accumulated_instance = np.where(curr_mask > 0, i + 1, accumulated_instance)
        
        if self.device != 'cpu':
            return torch.stack([torch.tensor(accumulated_mask).cuda(), torch.tensor(accumulated_instance).cuda()], dim=0)
        else:
            return torch.stack([torch.tensor(accumulated_mask), torch.tensor(accumulated_instance)], dim=0)

class E2E(nn.Module):
    """
    End-to-End model for image processing.
    """

    def __init__(self, num_classes, use_dense=True, attach_seg_head=True, train_bbox_decoder=True,
                 train_seg_decoder=False, device="cuda"):
        super(E2E, self).__init__()

        self.encoder = self._init_encoder()
        self.device = device
        self.detector = DetNet(num_classes, use_dense=CFG.use_dense, device=self.device)
        self.mask_head = self._init_mask_head()

        self.train_bbox_decoder = train_bbox_decoder
        self.train_seg_decoder = train_seg_decoder
        self.attach_seg_head = attach_seg_head
        self.features = []

        self._register_forward_hooks()
        self._freeze_encoder()
        self._freeze_mask_head()

    def _init_encoder(self):
        """
        Initialize the image encoder.
        """
        return ImageEncoderViT(
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

    def _init_mask_head(self):
        """
        Initialize the mask head.
        """
        prompt_embed_dim = 256
        image_size = 1024
        image_embedding_size = image_size // 16  # vit_patch_size = 16
        return MaskHead(image_embedding_size=image_embedding_size, image_size=image_size,
                        prompt_embed_dim=prompt_embed_dim, device=self.device)

    def _register_forward_hooks(self):
        """
        Register forward hooks for the encoder blocks.
        """
        for i, block in enumerate(self.encoder.blocks):
            block.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        Hook function to append output to features.
        """
        self.features.append(output)

    def _freeze_encoder(self):
        """
        Freeze the encoder parameters.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _freeze_mask_head(self):
        """
        Freeze the mask head parameters based on the training configuration.
        """
        prompt_encoder_params = self.mask_head.prompt_encoder.named_parameters()
        mask_decoder_params = self.mask_head.mask_decoder.named_parameters()
        detector_params = self.detector.named_parameters()

        if not self.train_seg_decoder and self.train_bbox_decoder:
            self._freeze_parameters(prompt_encoder_params, mask_decoder_params)
        elif not self.train_seg_decoder and not self.train_bbox_decoder:
            self._freeze_parameters(prompt_encoder_params, mask_decoder_params, detector_params)
        else:
            self._freeze_parameters(prompt_encoder_params, detector_params)

    @staticmethod
    def _freeze_parameters(*params):
        """
        Freeze the given parameters.
        """
        for parameters in params:
            for name, param in parameters:
                param.requires_grad = False

    def generate_point_grid(self,box_coord, num_points = 10):
        all_points = []
        for coord in box_coord:
            points = []
            x_min, y_min, x_max, y_max = coord
            x = int((x_max - x_min)/2)
            y = int((y_max - y_min)/2)
            points.append((x,y))
            all_points.append(points)
        return all_points


    def forward(self, inputs):
        self.features = []
        img_batch, annotations, batched_input = self._prepare_inputs(inputs)

        with torch.no_grad():
            out_ = self.encoder(img_batch)

        if self.train_bbox_decoder:
            self.detector.training = True
            return self.__forward_bbox(img_batch, self.features, annotations)
        else:
            self.detector.training = False
            return self._process_predictions(img_batch, out_, batched_input)

    def _prepare_inputs(self, inputs):
        annotations = None
        if self.training:
            img_batch, annotations = inputs
            annotations_de = self.process_proposals(copy.copy(annotations))
            batched_input = self._create_batched_input(img_batch, annotations_de)
        else:
            img_batch = inputs
            batched_input = self._create_batched_input(img_batch)

        img_batch = torch.stack([x["image"] for x in batched_input], dim=0)
        return img_batch, annotations, batched_input

    def _create_batched_input(self, img_batch, annotations_de=None):
        if annotations_de is None:
            return [{"image": img_batch[i], 'original_size': img_batch[i].shape[1:]} for i in range(img_batch.shape[0])]
        else:
            return [{"image": img_batch[i], "boxes": torch.tensor(annotations_de[i], device=self.device), 'original_size': img_batch[i].shape[1:]} for i in range(img_batch.shape[0])]

    def _process_predictions(self, img_batch, out_, batched_input):
        results = self.__forward_bbox(img_batch, self.features)
        no_pred, preds_collector, labels_collector, scores_collector = self._filter_results(results)

        if len(no_pred) != img_batch.shape[0]:
            return self._handle_predictions(img_batch, out_, batched_input, no_pred, preds_collector, labels_collector, scores_collector)
        elif not self.attach_seg_head:
            return preds_collector, labels_collector, scores_collector, no_pred
        else:
            return preds_collector, labels_collector, scores_collector, None, no_pred

    def _filter_results(self, results):
        no_pred = []
        preds_collector = []
        labels_collector = []
        scores_collector = []

        for index, result in enumerate(results):
            pred_boxes, label_list, scores_list = filter(*result)
            if not any(scores_list):
                no_pred.append(index)
                preds_collector.append([[]])
            else:
                preds_collector.append(self.process_proposals(pred_boxes))

            if self.attach_seg_head and not self.train_seg_decoder:
                if len(np.array(label_list).shape) == 1:
                    label_list = np.array(label_list)[np.newaxis, :]
            else:
                preds_collector.append(pred_boxes)

            labels_collector.append(label_list)
            scores_collector.append(scores_list)

        return no_pred, preds_collector, labels_collector, scores_collector

    def _handle_predictions(self, img_batch, out_, batched_input, no_pred, preds_collector, labels_collector, scores_collector):
        if self.attach_seg_head:
            if len(no_pred) > 0:
                out_ = self._remove_indices(out_, no_pred)

            batched_input = self._create_batched_input_mask(img_batch, preds_collector, labels_collector, no_pred)
            outputs = self.__forward_seg_head(batched_input, out_)

            if not self.train_seg_decoder:
                return preds_collector, labels_collector, scores_collector, outputs[0], outputs[1], no_pred
            else:
                return outputs
        else:
            return preds_collector, labels_collector, scores_collector

    def _create_batched_input_mask(self, img_batch, preds_collector, labels_collector, no_pred):
        return [{"labels": labels_collector[i][0], "boxes": torch.tensor(preds_collector[i][0], device=self.device), 'original_size': img_batch[i].shape[1:]} for i in range(img_batch.shape[0]) if i not in no_pred]


        
    def _remove_indices(self,tensor, indices):
        # Get the indices to keep
        keep_indices = [i for i in range(tensor.shape[0]) if i not in indices]

        # Remove the specified indices
        tensor = torch.index_select(tensor, dim=0, index=torch.tensor(keep_indices).to(self.device))

        return tensor
    def __forward_bbox(self,input,features,annotations=None):
        if annotations is not None:
            return self.detector(input,features,annotations)
        else: return self.detector(input,features)

    def __forward_seg_head(self,batched_input,image_embeddings):
         mask_logits = self.mask_head(batched_input,image_embeddings)
         return mask_logits
    def forward_seg_head(self,batched_input,image_embeddings):
        return self.__forward_seg_head(batched_input,image_embeddings)
        
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
        