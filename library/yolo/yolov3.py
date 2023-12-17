# Python Imports 
import os
import re
import time
import json
import random
import warnings
from pprint import pprint
from collections import Counter

# Library Imports
import cv2
from PIL import Image, ImageFile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom Imports

# Gloabal Variable/Settings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark  = True
ImageFile.LOAD_TRUNCATED_IMAGES = True

""" 
YOLOv3 architecture
"""
class YOLOv3CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class YOLOv3ResidualBlock(nn.Module):
    
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    YOLOv3CNNBlock(channels, channels // 2, kernel_size=1),
                    YOLOv3CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x

class YOLOv3ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            YOLOv3CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            YOLOv3CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    def __init__(self, device, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.device      = device
        self.layers      = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, YOLOv3ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x.to(self.device))

            if isinstance(layer, YOLOv3ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        """
            Information about architecture config:
                Tuple is structured by (filters, kernel_size, stride) 
                Every conv is a same convolution. 
                List is structured by "B" indicating a residual block followed by the number of repeats
                "S" is for scale prediction block and computing the yolo loss
                "U" is for upsampling the feature map and concatenating with a previous layer
        """
        yolov3_architecture = [
                                (32, 3, 1),
                                (64, 3, 2),
                                ["B", 1],
                                (128, 3, 2),
                                ["B", 2],
                                (256, 3, 2),
                                ["B", 8],
                                (512, 3, 2),
                                ["B", 8],
                                (1024, 3, 2),
                                ["B", 4],  # Till this point Model is Darknet-53
                                (512, 1, 1),
                                (1024, 3, 1),
                                "S",
                                (256, 1, 1),
                                "U",
                                (256, 1, 1),
                                (512, 3, 1),
                                "S",
                                (128, 1, 1),
                                "U",
                                (128, 1, 1),
                                (256, 3, 1),
                                "S",
                            ]
        for module in yolov3_architecture:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    YOLOv3CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(YOLOv3ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        YOLOv3ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        YOLOv3CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        YOLOv3ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

"""
YOLOv3 loss function
"""
class YOLOv3Loss(nn.Module):
    def __init__(self, box_format):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.box_format = box_format

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = yolov3_intersection_over_union(box_preds[obj], target[..., 1:5][obj], self.box_format).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

"""
YOLOv3 metrics & util modules
"""
def yolov3_seed_everything(seed=42):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False

def yolov3_get_mean_std(loader):
    
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum        += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum   += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches         += 1

    mean    = channels_sum / num_batches
    std     = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def yolov3_iou_width_height(boxes1, boxes2):
    
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    
    intersection    = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union           = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)
    
    return intersection / union

def yolov3_intersection_over_union(boxes_preds, boxes_labels, box_format):
    
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def yolov3_non_max_suppression(bboxes, iou_threshold, threshold, box_format):
    
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes              = [box for box in bboxes if box[1] > threshold]
    bboxes              = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms    = []

    while bboxes:
        
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or yolov3_intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def yolov3_mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, num_classes):
    
    """
    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections      = []
        ground_truths   = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP                  = torch.zeros((len(detections)))
        FP                  = torch.zeros((len(detections)))
        total_true_bboxes   = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = yolov3_intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum       = torch.cumsum(TP, dim=0)
        FP_cumsum       = torch.cumsum(FP, dim=0)
        recalls         = TP_cumsum / (total_true_bboxes + epsilon)
        precisions      = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions      = torch.cat((torch.tensor([1]), precisions))
        recalls         = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def yolov3_cells_to_bboxes(predictions, anchors, S, is_preds):
    
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    
    BATCH_SIZE      = predictions.shape[0]
    num_anchors     = len(anchors)
    box_predictions = predictions[..., 1:5]
    
    if is_preds:
        anchors                     = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2]   = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:]    = torch.exp(box_predictions[..., 2:]) * anchors
        scores                      = torch.sigmoid(predictions[..., 0:1])
        best_class                  = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores                      = predictions[..., 0:1]
        best_class                  = predictions[..., 5:6]

    cell_indices        = (torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device))
    x                   = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y                   = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h                 = 1 / S * box_predictions[..., 2:4]
    converted_bboxes    = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    
    return converted_bboxes.tolist()

def yolov3_get_class_accuracy(config, model, loader, log, threshold, log_mode):
    
    model.eval()
    tot_class_preds, correct_class  = 0, 0
    tot_noobj, correct_noobj        = 0, 0
    tot_obj, correct_obj            = 0, 0

    for idx, (x, y, image_path) in enumerate(tqdm(loader)):
        x = x.to(config["device"])
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i]    = y[i].to(config["device"])
            obj     = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj   = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class   += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj])
            tot_class_preds += torch.sum(obj)
            obj_preds        = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj     += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj         += torch.sum(obj)
            correct_noobj   += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj       += torch.sum(noobj)

    log.info(f"Class accuracy is     : {(correct_class/(tot_class_preds+1e-16))*100:2f}%", log_mode)
    log.info(f"No object accuracy is : {(correct_noobj/(tot_noobj+1e-16))*100:2f}%", log_mode)
    log.info(f"Object accuracy is    : {(correct_obj/(tot_obj+1e-16))*100:2f}%", log_mode)
    
    model.train()

def yolov3_save_checkpoint(model, optimizer, filename):
    
    checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                 }
    
    torch.save(checkpoint, filename)

def yolov3_load_checkpoint(config, checkpoint_file, model, optimizer, lr):

    checkpoint = torch.load(checkpoint_file, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

"""
YOLOv3 dataset transforms & dataloaders using the albumentations library
"""
def yolov3_transforms(config, log):

    IMAGE_SIZE  = config["dl"]["yolov3"]["image_size"]
    SCALE       = config["dl"]["yolov3"]["scale"]
    
    TRAIN_TRANSFORMS        = A.Compose(
                                        [
                                            A.LongestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),
                                            A.PadIfNeeded(
                                                min_height=int(IMAGE_SIZE * SCALE),
                                                min_width=int(IMAGE_SIZE * SCALE),
                                                border_mode=cv2.BORDER_CONSTANT,
                                            ),
                                            A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
                                            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                                            A.OneOf(
                                                [
                                                    A.ShiftScaleRotate(
                                                        rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                                                    ),
                                                    # A.IAAAffine(shear=15, p=0.5, mode="constant"),
                                                    # A.Affine(shear=15, p=0.5, mode="constant"),
                                                ],
                                                p=1.0,
                                            ),
                                            A.HorizontalFlip(p=0.5),
                                            A.Blur(p=0.1),
                                            A.CLAHE(p=0.1),
                                            A.Posterize(p=0.1),
                                            A.ToGray(p=0.1),
                                            A.ChannelShuffle(p=0.05),
                                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                                            ToTensorV2(),
                                        ],
                                        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
                                    )

    VALIDATION_TRANSFORMS   = A.Compose(
                                        [
                                            A.LongestMaxSize(max_size=IMAGE_SIZE),
                                            A.PadIfNeeded(
                                                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
                                            ),
                                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                                            ToTensorV2(),
                                        ],
                                        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
                                    )

    TEST_TRANSFORMS         = A.Compose(
                                        [
                                            A.LongestMaxSize(max_size=IMAGE_SIZE),
                                            A.PadIfNeeded(
                                                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
                                            ),
                                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                                            ToTensorV2(),
                                        ]
                                    )

    INFERENCE_TRANSFORMS    = A.Compose(
                                        [
                                            A.LongestMaxSize(max_size=IMAGE_SIZE),
                                            A.PadIfNeeded(
                                                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
                                            ),
                                            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                                            ToTensorV2(),
                                        ]
                                    )

    return TRAIN_TRANSFORMS, VALIDATION_TRANSFORMS, TEST_TRANSFORMS, INFERENCE_TRANSFORMS

def yolov3_loaders(config, dataset_name, loader_type, log, infer_dir=None):

    TRAIN_TRANSFORMS, VALIDATION_TRANSFORMS, TEST_TRANSFORMS, INFERENCE_TRANSFORMS = yolov3_transforms(config, log)

    # IMAGE_SIZE              = config["dl"]["yolov3"]["image_size"]
    ANCHORS                 = config["dl"]["yolov3"]["anchors"]
    BATCH_SIZE              = config["dl"]["yolov3"]["batch_size"]
    NUM_WORKERS             = config["dl"]["yolov3"]["workers"]
    PIN_MEMORY              = config["dl"]["yolov3"]["pin_memory"]
    SCALES                  = config["dl"]["yolov3"]["scales"]
    TRAIN_IMG_DIR           = os.path.join(config["datasets"][dataset_name]["train_dir"], config["datasets"][dataset_name]["tile_dir"])
    TRAIN_LABEL_DIR         = os.path.join(config["datasets"][dataset_name]["train_dir"], config["datasets"][dataset_name]["tile_dir"])
    TEST_IMG_DIR            = os.path.join(config["datasets"][dataset_name]["test_dir"], config["datasets"][dataset_name]["tile_dir"])
    TEST_LABEL_DIR          = os.path.join(config["datasets"][dataset_name]["test_dir"], config["datasets"][dataset_name]["tile_dir"])
    VALIDATION_IMG_DIR      = os.path.join(config["datasets"][dataset_name]["validation_dir"], config["datasets"][dataset_name]["tile_dir"])
    VALIDATION_LABEL_DIR    = os.path.join(config["datasets"][dataset_name]["validation_dir"], config["datasets"][dataset_name]["tile_dir"])
    TRAIN_CSV_PATH          = os.path.join(config["datasets"][dataset_name]["base_dir"], config["datasets"][dataset_name]["train_csv"])
    TEST_CSV_PATH           = os.path.join(config["datasets"][dataset_name]["base_dir"], config["datasets"][dataset_name]["test_csv"])
    VALIDATION_CSV_PATH     = os.path.join(config["datasets"][dataset_name]["base_dir"], config["datasets"][dataset_name]["validation_csv"])
    INFERENCE_DIR           = os.path.join(config["datasets"][dataset_name]["inference_dir"], config["datasets"][dataset_name]["tile_dir"])
    TEST_DIR                = os.path.join(config["datasets"][dataset_name]["test_dir"], config["datasets"][dataset_name]["tile_dir"])
    UNSEEN_DIR              = config["datasets"][dataset_name]["unseen_dir"]

    if infer_dir is not None:
        INFERENCE_DIR = os.path.join(infer_dir, config["datasets"][dataset_name]["tile_dir"])

    if loader_type=="train":
        # Train dataset & dataloader
        dataset       = YOLOv3TrainValidation(TRAIN_CSV_PATH, 
                                                        transform=TRAIN_TRANSFORMS,
                                                        S=SCALES,
                                                        img_dir=TRAIN_IMG_DIR,
                                                        label_dir=TRAIN_LABEL_DIR,
                                                        anchors=ANCHORS)
        loader        = DataLoader(dataset=dataset, 
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        shuffle=True,
                                        drop_last=False)
    
    if loader_type=="validation":
        # Validation dataset & dataloader
        dataset  = YOLOv3TrainValidation(VALIDATION_CSV_PATH,
                                                        transform=VALIDATION_TRANSFORMS,
                                                        S=SCALES,
                                                        img_dir=VALIDATION_IMG_DIR,
                                                        label_dir=VALIDATION_LABEL_DIR,
                                                        anchors=ANCHORS)
        loader   = DataLoader(dataset=dataset, 
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        shuffle=False,
                                        drop_last=False)
    
    if loader_type=="test": 
        # Test dataset & dataloader    
        dataset        = YOLOv3Inference(image_dir=TEST_DIR,
                                         transform=TEST_TRANSFORMS,
                                         S=SCALES,
                                         anchors=ANCHORS,
                                         exts=config["files"]["allowed_image_extensions"])
        loader         = DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        shuffle=False,
                                        drop_last=False)
    
    if loader_type=="inference": 
        # Inference dataset & dataloader     
        dataset   = YOLOv3Inference(image_dir=INFERENCE_DIR,
                                    transform=INFERENCE_TRANSFORMS,
                                    S=SCALES,
                                    anchors=ANCHORS,
                                    exts=config["files"]["allowed_image_extensions"])
        loader    = DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        shuffle=False,
                                        drop_last=False)

    if loader_type=="unseen": 
        # Unseen dataset & dataloader     
        dataset   = YOLOv3Inference(image_dir=UNSEEN_DIR,
                                    transform=INFERENCE_TRANSFORMS,
                                    S=SCALES,
                                    anchors=ANCHORS,
                                    exts=config["files"]["allowed_image_extensions"])
        loader    = DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=PIN_MEMORY,
                                        shuffle=False,
                                        drop_last=False)
    
    return loader

"""
YOLOv3 custom class for train, validation, test & inference datasets
"""
class YOLOv3TrainValidation(Dataset):
    
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        
        self.annotations            = pd.read_csv(csv_file)
        self.img_dir                = img_dir
        self.label_dir              = label_dir
        self.image_size             = image_size
        self.transform              = transform
        self.S                      = S
        self.anchors                = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors            = self.anchors.shape[0]
        self.num_anchors_per_scale  = self.num_anchors // 3
        self.C                      = C
        self.ignore_iou_thresh      = 0.5

    def __len__(self):
        
        return len(self.annotations)

    def __getitem__(self, index):
        
        label_path  = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes      = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path    = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image       = np.array(Image.open(img_path).convert("RGB"))

        # albumegations
        if self.transform:
            augmentations   = self.transform(image=image, bboxes=bboxes)
            image           = augmentations["image"]
            bboxes          = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:           
            iou_anchors                         = yolov3_iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices                      = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label    = box
            has_anchor                          = [False] * 3  # each scale should have one anchor
            
            for anchor_idx in anchor_indices:
                scale_idx       = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S               = self.S[scale_idx]
                i, j            = int(S * y), int(S * x)  # which cell
                anchor_taken    = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0]    = 1
                    x_cell, y_cell                                  = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell                         = (width * S, height * S)  # can be greater than 1 since it's relative to cell
                    box_coordinates                                 = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5]  = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5]    = int(class_label)
                    has_anchor[scale_idx]                           = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets), img_path

class YOLOv3Test(Dataset):
    
    def __init__(self, image_dir, anchors, exts, image_size=416, S=[13, 26, 52], C=20, transform=None):

        self.images                 = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(tuple(exts))]
        self.image_size             = image_size
        self.transform              = transform
        self.S                      = S
        self.anchors                = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors            = self.anchors.shape[0]
        self.num_anchors_per_scale  = self.num_anchors // 3
        self.C                      = C
        self.ignore_iou_thresh      = 0.5

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.transform:
            augmentations   = self.transform(image=image)
            image           = augmentations["image"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        return image, tuple(targets), self.images[idx]

class YOLOv3Inference(Dataset):
    
    def __init__(self, image_dir, anchors, exts, image_size=416, S=[13, 26, 52], C=20, transform=None):

        self.images                 = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(tuple(exts))]
        self.image_size             = image_size
        self.transform              = transform
        self.S                      = S
        self.anchors                = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors            = self.anchors.shape[0]
        self.num_anchors_per_scale  = self.num_anchors // 3
        self.C                      = C
        self.ignore_iou_thresh      = 0.5

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        if self.transform:
            augmentations   = self.transform(image=image)
            image           = augmentations["image"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        return image, tuple(targets), self.images[idx]

"""
YOLOv3 train, validation & test modules
"""
def yolov3_epoch(config, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    
    loop    = tqdm(train_loader, leave=True)
    losses  = []
    
    for batch_idx, (x, y, img_path) in enumerate(loop):
        x           = x.to(config["device"])
        y0, y1, y2  = (y[0].to(config["device"]),y[1].to(config["device"]),y[2].to(config["device"]))

        with torch.cuda.amp.autocast():
            out     = model(x)
            loss    = (loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1]) + loss_fn(out[2], y2, scaled_anchors[2]))

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def yolov3_get_validation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device):
    
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx       = 0
    all_pred_boxes  = []
    all_true_boxes  = []

    for batch_idx, (x, labels, img_path) in enumerate(tqdm(loader)):
        
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size  = x.shape[0]
        bboxes      = [[] for _ in range(batch_size)]

        for i in range(3):
            S               = predictions[i].shape[2]
            anchor          = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i   = yolov3_cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = yolov3_cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):

            nms_boxes = yolov3_non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
            
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    
    return all_pred_boxes, all_true_boxes

def yolov3_train(config, dataset_name, log):

    train_start_time = time.time()
    
    log.info("Starting YOLOv3 training for given dataset...", task="train")
    
    log.info("Initializing model...", task="train")
    model       = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])
    
    log.info("Initializing optimizer...", task="train")
    optimizer   = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"])

    log.info("Initializing loss, scaler...", task="train")
    loss_fn     = YOLOv3Loss(box_format=config["dl"]["yolov3"]["box_format"])
    scaler      = torch.cuda.amp.GradScaler()
    
    log.info("Preparing PyTorch Dataloaders for train, validation and test datasets...", task="train")
    train_loader        = yolov3_loaders(config, dataset_name, "train", log)
    validation_loader   = yolov3_loaders(config, dataset_name, "validation", log)

    log.info("Load existing weights...", task="train")
    if config["dl"]["yolov3"]["load_model"]:
        yolov3_load_checkpoint(config, config["datasets"][dataset_name]["load_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

    scaled_anchors = (torch.tensor(config["dl"]["yolov3"]["anchors"]) * torch.tensor(config["dl"]["yolov3"]["scales"]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config["device"])
    
    for epoch in range(config["dl"]["yolov3"]["epochs"]):

        epoch_start_time = time.time()
        
        log.info("Starting epoch..."+str(epoch), task="train")
        yolov3_epoch(config, train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        
        log.info("Saving intermediate checkpoint/weights...", task="train")
        if config["dl"]["yolov3"]["save_model"]:
            yolov3_save_checkpoint(model, optimizer, filename=config["datasets"][dataset_name]["save_checkpoint_file"])

        if config["dl"]["yolov3"]["load_model"]:
            yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

        yolov3_get_class_accuracy(config, model, validation_loader, log, threshold=config["dl"]["yolov3"]["confidence_threshold"], log_mode="train")

        pred_boxes, true_boxes  = yolov3_get_validation_bboxes(validation_loader, 
                                                               model, 
                                                               iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"],
                                                               anchors=config["dl"]["yolov3"]["anchors"], 
                                                               threshold=config["dl"]["yolov3"]["confidence_threshold"],
                                                               box_format=config["dl"]["yolov3"]["box_format"],
                                                               device=config["device"])
        
        mapval                  = yolov3_mean_average_precision(pred_boxes, 
                                                                true_boxes,
                                                                iou_threshold=config["dl"]["yolov3"]["mAP_iou_threshold"],
                                                                box_format=config["dl"]["yolov3"]["box_format"],
                                                                num_classes=len(config["datasets"][dataset_name]["classes"]))  
        
        log.info("mAP for epoch..."+str(epoch)+"..."+str(mapval.item()), task="train")

        log.info("Saving intermediate checkpoint/weights...with mAP threshold", task="train")
        if round(mapval.item(), 2)>0.4:
            yolov3_save_checkpoint(model, optimizer, filename=config["datasets"][dataset_name]["save_checkpoint_file"].replace(".pth","-epoch-"+str(epoch)+"-mAP-"+str(int(round(mapval.item(), 2)*100))+".pth"))

        model.train()
    
        epoch_elapsed_time = time.time() - epoch_start_time
        log.info("Ending Epoch..."+str(epoch)+", execution time: "+ str(time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))+" HH:MM:SS ", task="train")

    train_elapsed_time = time.time() - train_start_time
    log.info("Training...execution time: "+ str(time.strftime("%H:%M:%S", time.gmtime(train_elapsed_time)))+" HH:MM:SS ", task="train")

# def yolov3_get_test_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device):
    
#     # make sure model is in eval before get bboxes
#     model.eval()
#     train_idx       = 0
#     all_pred_boxes  = []
#     all_true_boxes  = []
#     test_pred_bboxes  = {}

#     for batch_idx, (x, labels, batch_image_names) in enumerate(tqdm(loader)):
        
#         x = x.to(device)

#         with torch.no_grad():
#             predictions = model(x)

#         batch_size  = x.shape[0]
#         bboxes      = [[] for _ in range(batch_size)]

#         for i in range(3):
#             S               = predictions[i].shape[2]
#             anchor          = torch.tensor([*anchors[i]]).to(device) * S
#             boxes_scale_i   = yolov3_cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
#             for idx, (box) in enumerate(boxes_scale_i):
#                 bboxes[idx] += box

#         # we just want one bbox for each label, not one for each scale
#         true_bboxes = yolov3_cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

#         for idx in range(batch_size):
            
#             image_pred_bboxes = []

#             nms_boxes = yolov3_non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
            
#             for nms_box in nms_boxes:
#                 all_pred_boxes.append([train_idx] + nms_box)
#                 image_pred_bboxes.append([train_idx] + nms_box)

#             for box in true_bboxes[idx]:
#                 if box[1] > threshold:
#                     all_true_boxes.append([train_idx] + box)

#             train_idx += 1

#             test_pred_bboxes[batch_image_names[idx]] = image_pred_bboxes

#     model.train()
    
#     return all_pred_boxes, all_true_boxes, test_pred_bboxes

# def yolov3_test(config, dataset_name, test_image, log):
    
#     log.info("Starting YOLOv3 testing for given dataset...", task="test")
    
#     log.info("Initializing model...", task="test")
#     model     = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])
    
#     log.info("Initializing optimizer...", task="train")
#     optimizer = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"]) 
    
#     log.info("Load existing weights...", task="test")
#     if config["dl"]["yolov3"]["load_model"]:
#         yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

#     log.info("Loading inference dataset & dataloader...", task="test")
#     test_loader = yolov3_loaders(config, dataset_name, "test", log)

#     # yolov3_get_class_accuracy(config, model, test_loader, log, threshold=config["dl"]["yolov3"]["confidence_threshold"], log_mode="test")

#     log.info("Predicted bboxes on test dataset...", task="test")
#     pred_boxes, true_boxes, test_pred_bboxes = yolov3_get_test_bboxes(test_loader,
#                                                                       model, 
#                                                                       iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"], 
#                                                                       anchors=config["dl"]["yolov3"]["anchors"],
#                                                                       threshold=config["dl"]["yolov3"]["confidence_threshold"],
#                                                                       box_format=config["dl"]["yolov3"]["box_format"],
#                                                                       device=config["device"])
#     # mapval                 = yolov3_mean_average_precision(pred_boxes, 
#     #                                                        true_boxes,
#     #                                                        iou_threshold=config["dl"]["yolov3"]["mAP_iou_threshold"],
#     #                                                        box_format=config["dl"]["yolov3"]["box_format"],
#     #                                                        num_classes=len(config["datasets"][dataset_name]["classes"]))  

#     # log.info("mAP for image..."+str(mapval.item()), task="test")
    
#     log.info("Consolidating tiles bboxes to original image bboxes...", task="test")
#     test_image_png = yolov3_combine_tile_pred_bboxes(config, test_image, test_pred_bboxes, config["datasets"][dataset_name]["classes"])

#     log.info("Completed testing...", task="test")
    
#     log.info("Removing temp png image...", task="inference")
#     os.remove(test_image_png)

"""
YOLOv3 inference modules
"""
def yolov3_inference(config, dataset_name, inference_image, log):

    def yolov3_get_tile_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device):
        
        # make sure model is in eval before get bboxes
        model.eval()
        train_idx           = 0
        images_pred_bboxes  = {}
        # all_pred_boxes      = []

        for batch_idx, (x, labels, batch_image_names) in enumerate(tqdm(loader)):

            x = x.to(device)

            with torch.no_grad():
                predictions = model(x)

            batch_size  = x.shape[0]
            bboxes      = [[] for _ in range(batch_size)]

            for i in range(3):
                S               = predictions[i].shape[2]
                anchor          = torch.tensor([*anchors[i]]).to(device) * S
                boxes_scale_i   = yolov3_cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            for idx in range(batch_size):
                image_pred_bboxes = []    
                nms_boxes = yolov3_non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
                
                for nms_box in nms_boxes:
                    # all_pred_boxes.append([train_idx] + nms_box)
                    image_pred_bboxes.append([train_idx] + nms_box)

                images_pred_bboxes[batch_image_names[idx]] = image_pred_bboxes

                train_idx += 1

        model.train()

        return images_pred_bboxes

    def yolov3_get_image_bboxes(config, dataset_name, inference_image, images_pred_bboxes, labels, log, draw_bbox):
        
        """Plots predicted bounding boxes on the image"""
        inference_image      = inference_image.replace(".tif", ".png")
        inference_image_bbox = inference_image.replace(".png", "_bboxes.png")
        inference_bboxes     = []
        
        class_labels        = labels 

        cv2im = cv2.imread(inference_image)

        for image, bboxes in images_pred_bboxes.items():

            im                  = np.array(Image.open(image).convert("RGB"))
            height, width, _    = im.shape

            local_tile = { 
                "tile_ul": (0, 0), 
                "tile_ur": (416, 0), 
                "tile_br": (416, 416), 
                "tile_bl": (0, 416)
            }

            # find the bbox in image coordinates by translating from tile coordinate to respective full image coordinate 
            for bbox in bboxes:
                assert len(bbox) == 7, "box should contain image index, class pred, confidence, x, y, width, height"
                class_pred      = bbox[1]
                class_prob      = round(bbox[2]*100, 2)
                bbox            = bbox[3:]
                # After slicing
                # box[0] is x midpoint, box[1] is y midpoint, box[2] is width, box[3] is height
                
                # ul
                upper_left_x    = bbox[0] - bbox[2] / 2
                upper_left_y    = bbox[1] - bbox[3] / 2
                
                # ur
                upper_right_x   = (bbox[0] - bbox[2] / 2)+bbox[2]
                upper_right_y   = bbox[1] - bbox[3] / 2

                # br
                bottom_right_x   = bbox[0] + bbox[2] / 2
                bottom_right_y   = bbox[1] + bbox[3] / 2

                # bl
                bottom_left_x   = bbox[0] - bbox[2] / 2
                bottom_left_y   = (bbox[1] - bbox[3] / 2)+bbox[3]
                
                
                # denormalized bbox 
                upper_left_x    = int(upper_left_x * width)
                upper_left_y    = int(upper_left_y * height)

                upper_right_x    = int(upper_right_x * width)
                upper_right_y    = int(upper_right_y * height)
                
                bottom_right_x   = int(bottom_right_x * width)
                bottom_right_y   = int(bottom_right_y * height)

                bottom_left_x   = int(bottom_left_x * width)
                bottom_left_y   = int(bottom_left_y * height)

                bbox_width      = int(bbox[2] * width)
                bbox_height     = int(bbox[3] * height)

                local_bbox = {
                    "bbox_ul"   : (upper_left_x, upper_left_y),
                    "bbox_ur"   : (upper_right_x, upper_right_y),
                    "bbox_br"   : (bottom_right_x, bottom_right_y),
                    "bbox_bl"   : (bottom_left_x, bottom_left_y)
                }

                image_tile_ul_from_name = tuple(re.findall(r"\d{1,5}\-\d{1,5}", image)[0].split("-"))

                image_tile_ul_from_name = tuple((int(image_tile_ul_from_name[0]), int(image_tile_ul_from_name[1])))

                image_tile = {
                    "tile_ul": (local_tile["tile_ul"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ul"][1]+image_tile_ul_from_name[1]), 
                    "tile_ur": (local_tile["tile_ur"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ur"][1]+image_tile_ul_from_name[1]), 
                    "tile_br": (local_tile["tile_br"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_br"][1]+image_tile_ul_from_name[1]), 
                    "tile_bl": (local_tile["tile_bl"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_bl"][1]+image_tile_ul_from_name[1])
                }

                image_bbox = {
                    "bbox_ul": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0], 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_ur": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width,
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_br": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width, 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                    "bbox_bl": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0],
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                }

                image_bbox["confidence"]    = class_prob
                image_bbox["class_id"]      = int(class_pred)   
                image_bbox["class_name"]    = class_labels[int(class_pred)]

                inference_bboxes.append(image_bbox)

                if draw_bbox:
                    if int(class_prob)>config["datasets"][dataset_name]["inference_confidence"]:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["bbox_color"], config["opencv"]["bbox_thickness"])
                    else:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["alert_color"], config["opencv"]["bbox_thickness"])
        
        if draw_bbox:
            cv2.imwrite(inference_image_bbox, cv2im)

        return inference_image, inference_image_bbox, inference_bboxes

    log.info("Initializing model...", task="inference")
    model     = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])

    log.info("Initializing optimizer...", task="inference")
    optimizer = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"]) 
    
    log.info("Load existing weights...", task="inference")
    yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

    log.info("Loading inference dataset & dataloader...", task="inference")
    inference_loader = yolov3_loaders(config, dataset_name, "inference", log)

    log.info("Predicted bboxes...", task="inference")
    images_pred_bboxes = yolov3_get_tile_bboxes(inference_loader, 
                                                     model, 
                                                     iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"], 
                                                     anchors=config["dl"]["yolov3"]["anchors"],
                                                     threshold=config["dl"]["yolov3"]["confidence_threshold"],
                                                     box_format=config["dl"]["yolov3"]["box_format"],
                                                     device=config["device"])

    log.info("Consolidating tiles bboxes to original image bboxes...", task="inference")
    inference_image_png, inference_image_bbox, inference_bboxes = yolov3_get_image_bboxes(config, 
                                                                                dataset_name, 
                                                                                inference_image, 
                                                                                images_pred_bboxes, 
                                                                                config["datasets"][dataset_name]["classes"],
                                                                                log,
                                                                                draw_bbox=True)

    log.info("Completed inference...", task="inference")

    log.info("Removing temp png image...", task="inference")
    os.remove(inference_image_png)

    return inference_image_png, inference_image_bbox, inference_bboxes

def yolov3_inference_api(config, dataset_name, inference_image, inference_dir, log):

    def yolov3_get_tile_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device):
        
        # make sure model is in eval before get bboxes
        model.eval()
        train_idx           = 0
        images_pred_bboxes  = {}
        # all_pred_boxes      = []

        for batch_idx, (x, labels, batch_image_names) in enumerate(tqdm(loader)):

            x = x.to(device)

            with torch.no_grad():
                predictions = model(x)

            batch_size  = x.shape[0]
            bboxes      = [[] for _ in range(batch_size)]

            for i in range(3):
                S               = predictions[i].shape[2]
                anchor          = torch.tensor([*anchors[i]]).to(device) * S
                boxes_scale_i   = yolov3_cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            for idx in range(batch_size):
                image_pred_bboxes = []    
                nms_boxes = yolov3_non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
                
                for nms_box in nms_boxes:
                    # all_pred_boxes.append([train_idx] + nms_box)
                    image_pred_bboxes.append([train_idx] + nms_box)

                images_pred_bboxes[batch_image_names[idx]] = image_pred_bboxes

                train_idx += 1

        model.train()

        return images_pred_bboxes

    def yolov3_get_image_bboxes(config, dataset_name, inference_image, images_pred_bboxes, labels, log, draw_bbox):
        
        """Plots predicted bounding boxes on the image"""
        inference_image      = inference_image.replace(".tif", ".png")
        inference_image_bbox = inference_image.replace(".png", "_bboxes.png")
        inference_bboxes     = []
        
        class_labels        = labels 

        cv2im = cv2.imread(inference_image)

        for image, bboxes in images_pred_bboxes.items():

            im                  = np.array(Image.open(image).convert("RGB"))
            height, width, _    = im.shape

            local_tile = { 
                "tile_ul": (0, 0), 
                "tile_ur": (416, 0), 
                "tile_br": (416, 416), 
                "tile_bl": (0, 416)
            }

            # find the bbox in image coordinates by translating from tile coordinate to respective full image coordinate 
            for bbox in bboxes:
                assert len(bbox) == 7, "box should contain image index, class pred, confidence, x, y, width, height"
                class_pred      = bbox[1]
                class_prob      = round(bbox[2]*100, 2)
                bbox            = bbox[3:]
                # After slicing
                # box[0] is x midpoint, box[1] is y midpoint, box[2] is width, box[3] is height
                
                # ul
                upper_left_x    = bbox[0] - bbox[2] / 2
                upper_left_y    = bbox[1] - bbox[3] / 2
                
                # ur
                upper_right_x   = (bbox[0] - bbox[2] / 2)+bbox[2]
                upper_right_y   = bbox[1] - bbox[3] / 2

                # br
                bottom_right_x   = bbox[0] + bbox[2] / 2
                bottom_right_y   = bbox[1] + bbox[3] / 2

                # bl
                bottom_left_x   = bbox[0] - bbox[2] / 2
                bottom_left_y   = (bbox[1] - bbox[3] / 2)+bbox[3]
                
                
                # denormalized bbox 
                upper_left_x    = int(upper_left_x * width)
                upper_left_y    = int(upper_left_y * height)

                upper_right_x    = int(upper_right_x * width)
                upper_right_y    = int(upper_right_y * height)
                
                bottom_right_x   = int(bottom_right_x * width)
                bottom_right_y   = int(bottom_right_y * height)

                bottom_left_x   = int(bottom_left_x * width)
                bottom_left_y   = int(bottom_left_y * height)

                bbox_width      = int(bbox[2] * width)
                bbox_height     = int(bbox[3] * height)

                local_bbox = {
                    "bbox_ul"   : (upper_left_x, upper_left_y),
                    "bbox_ur"   : (upper_right_x, upper_right_y),
                    "bbox_br"   : (bottom_right_x, bottom_right_y),
                    "bbox_bl"   : (bottom_left_x, bottom_left_y)
                }

                image_tile_ul_from_name = tuple(re.findall(r"\d{1,5}\-\d{1,5}", image)[0].split("-"))

                image_tile_ul_from_name = tuple((int(image_tile_ul_from_name[0]), int(image_tile_ul_from_name[1])))

                image_tile = {
                    "tile_ul": (local_tile["tile_ul"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ul"][1]+image_tile_ul_from_name[1]), 
                    "tile_ur": (local_tile["tile_ur"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ur"][1]+image_tile_ul_from_name[1]), 
                    "tile_br": (local_tile["tile_br"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_br"][1]+image_tile_ul_from_name[1]), 
                    "tile_bl": (local_tile["tile_bl"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_bl"][1]+image_tile_ul_from_name[1])
                }

                image_bbox = {
                    "bbox_ul": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0], 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_ur": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width,
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_br": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width, 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                    "bbox_bl": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0],
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                }

                image_bbox["confidence"]    = class_prob
                image_bbox["class_id"]      = int(class_pred)   
                image_bbox["class_name"]    = class_labels[int(class_pred)]

                inference_bboxes.append(image_bbox)

                if draw_bbox:
                    if int(class_prob)>config["datasets"][dataset_name]["inference_confidence"]:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["bbox_color"], config["opencv"]["bbox_thickness"])
                    else:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["alert_color"], config["opencv"]["bbox_thickness"])
        
        if draw_bbox:
            cv2.imwrite(inference_image_bbox, cv2im)

        return inference_image, inference_image_bbox, inference_bboxes

    log.info("Initializing model...", task="inference")
    model     = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])

    log.info("Initializing optimizer...", task="inference")
    optimizer = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"]) 
    
    log.info("Load existing weights...", task="inference")
    yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

    log.info("Loading inference dataset & dataloader...", task="inference")
    inference_loader = yolov3_loaders(config, dataset_name, "inference", log, infer_dir=inference_dir)

    log.info("Predicted bboxes...", task="inference")
    images_pred_bboxes = yolov3_get_tile_bboxes(inference_loader, 
                                                     model, 
                                                     iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"], 
                                                     anchors=config["dl"]["yolov3"]["anchors"],
                                                     threshold=config["dl"]["yolov3"]["confidence_threshold"],
                                                     box_format=config["dl"]["yolov3"]["box_format"],
                                                     device=config["device"])

    log.info("Consolidating tiles bboxes to original image bboxes...", task="inference")
    inference_image_png, inference_image_bbox, inference_bboxes = yolov3_get_image_bboxes(config, 
                                                                                dataset_name, 
                                                                                inference_image, 
                                                                                images_pred_bboxes, 
                                                                                config["datasets"][dataset_name]["classes"],
                                                                                log,
                                                                                draw_bbox=True)

    log.info("Completed inference...", task="inference")

    log.info("Removing temp png image...", task="inference")
    os.remove(inference_image_png)

    return inference_image_png, inference_image_bbox, inference_bboxes

def yolov3_inference_statistics(config, model_filename, dataset_name, inference_image, log):

    def yolov3_get_tile_bboxes(loader, model, iou_threshold, anchors, threshold, box_format, device):
        
        # make sure model is in eval before get bboxes
        model.eval()
        train_idx           = 0
        images_pred_bboxes  = {}
        # all_pred_boxes      = []

        for batch_idx, (x, labels, batch_image_names) in enumerate(tqdm(loader)):

            x = x.to(device)

            with torch.no_grad():
                predictions = model(x)

            batch_size  = x.shape[0]
            bboxes      = [[] for _ in range(batch_size)]

            for i in range(3):
                S               = predictions[i].shape[2]
                anchor          = torch.tensor([*anchors[i]]).to(device) * S
                boxes_scale_i   = yolov3_cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

            for idx in range(batch_size):
                image_pred_bboxes = []    
                nms_boxes = yolov3_non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)
                
                for nms_box in nms_boxes:
                    # all_pred_boxes.append([train_idx] + nms_box)
                    image_pred_bboxes.append([train_idx] + nms_box)

                images_pred_bboxes[batch_image_names[idx]] = image_pred_bboxes

                train_idx += 1

        model.train()

        return images_pred_bboxes

    def yolov3_get_image_bboxes(config, model_filename, dataset_name, inference_image, images_pred_bboxes, labels, log, draw_bbox):
        
        """Plots predicted bounding boxes on the image"""
        inference_image      = inference_image.replace(".tif", ".png")
        inference_image_bbox = inference_image.replace(".png", "_"+model_filename+"_bboxes.png")
        inference_bboxes     = []
        
        class_labels        = labels 

        cv2im = cv2.imread(inference_image)

        for image, bboxes in images_pred_bboxes.items():

            im                  = np.array(Image.open(image).convert("RGB"))
            height, width, _    = im.shape

            local_tile = { 
                "tile_ul": (0, 0), 
                "tile_ur": (416, 0), 
                "tile_br": (416, 416), 
                "tile_bl": (0, 416)
            }

            # find the bbox in image coordinates by translating from tile coordinate to respective full image coordinate 
            for bbox in bboxes:
                assert len(bbox) == 7, "box should contain image index, class pred, confidence, x, y, width, height"
                class_pred      = bbox[1]
                class_prob      = round(bbox[2]*100, 2)
                bbox            = bbox[3:]
                # After slicing
                # box[0] is x midpoint, box[1] is y midpoint, box[2] is width, box[3] is height
                
                # ul
                upper_left_x    = bbox[0] - bbox[2] / 2
                upper_left_y    = bbox[1] - bbox[3] / 2
                
                # ur
                upper_right_x   = (bbox[0] - bbox[2] / 2)+bbox[2]
                upper_right_y   = bbox[1] - bbox[3] / 2

                # br
                bottom_right_x   = bbox[0] + bbox[2] / 2
                bottom_right_y   = bbox[1] + bbox[3] / 2

                # bl
                bottom_left_x   = bbox[0] - bbox[2] / 2
                bottom_left_y   = (bbox[1] - bbox[3] / 2)+bbox[3]
                
                
                # denormalized bbox 
                upper_left_x    = int(upper_left_x * width)
                upper_left_y    = int(upper_left_y * height)

                upper_right_x    = int(upper_right_x * width)
                upper_right_y    = int(upper_right_y * height)
                
                bottom_right_x   = int(bottom_right_x * width)
                bottom_right_y   = int(bottom_right_y * height)

                bottom_left_x   = int(bottom_left_x * width)
                bottom_left_y   = int(bottom_left_y * height)

                bbox_width      = int(bbox[2] * width)
                bbox_height     = int(bbox[3] * height)

                local_bbox = {
                    "bbox_ul"   : (upper_left_x, upper_left_y),
                    "bbox_ur"   : (upper_right_x, upper_right_y),
                    "bbox_br"   : (bottom_right_x, bottom_right_y),
                    "bbox_bl"   : (bottom_left_x, bottom_left_y)
                }

                image_tile_ul_from_name = tuple(re.findall(r"\d{1,5}\-\d{1,5}", image)[0].split("-"))

                image_tile_ul_from_name = tuple((int(image_tile_ul_from_name[0]), int(image_tile_ul_from_name[1])))

                image_tile = {
                    "tile_ul": (local_tile["tile_ul"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ul"][1]+image_tile_ul_from_name[1]), 
                    "tile_ur": (local_tile["tile_ur"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_ur"][1]+image_tile_ul_from_name[1]), 
                    "tile_br": (local_tile["tile_br"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_br"][1]+image_tile_ul_from_name[1]), 
                    "tile_bl": (local_tile["tile_bl"][0]+image_tile_ul_from_name[0], 
                                local_tile["tile_bl"][1]+image_tile_ul_from_name[1])
                }

                image_bbox = {
                    "bbox_ul": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0], 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_ur": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width,
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]),
                    "bbox_br": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0]+bbox_width, 
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                    "bbox_bl": (local_bbox["bbox_ul"][0]+image_tile["tile_ul"][0],
                                local_bbox["bbox_ul"][1]+image_tile["tile_ul"][1]+bbox_height),
                }

                image_bbox["confidence"]    = class_prob
                image_bbox["class_id"]      = int(class_pred)   
                image_bbox["class_name"]    = class_labels[int(class_pred)]

                inference_bboxes.append(image_bbox)

                if draw_bbox:
                    if int(class_prob)>config["datasets"][dataset_name]["inference_confidence"]:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["bbox_color"], config["opencv"]["bbox_thickness"])
                    else:
                        cv2im = cv2.rectangle(cv2im, image_bbox["bbox_ul"], image_bbox["bbox_br"], config["opencv"]["alert_color"], config["opencv"]["bbox_thickness"])
        
        if draw_bbox:
            cv2.imwrite(inference_image_bbox, cv2im)

        return inference_image, inference_image_bbox, inference_bboxes

    log.info("Initializing model...", task="inference")
    model     = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])

    log.info("Initializing optimizer...", task="inference")
    optimizer = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"]) 
    
    log.info("Load existing weights...", task="inference")
    yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

    # model_filename = os.path.basename(config["datasets"][dataset_name]["save_checkpoint_file"]).replace(".pth","")

    log.info("Loading inference dataset & dataloader...", task="inference")
    inference_loader = yolov3_loaders(config, dataset_name, "inference", log)

    log.info("Predicted bboxes...", task="inference")
    images_pred_bboxes = yolov3_get_tile_bboxes(inference_loader, 
                                                model,
                                                iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"], 
                                                anchors=config["dl"]["yolov3"]["anchors"],
                                                threshold=config["dl"]["yolov3"]["confidence_threshold"],
                                                box_format=config["dl"]["yolov3"]["box_format"],
                                                device=config["device"])

    log.info("Consolidating tiles bboxes to original image bboxes...", task="inference")
    inference_image_png, inference_image_bbox, inference_bboxes = yolov3_get_image_bboxes(config,
                                                                                          model_filename,
                                                                                          dataset_name, 
                                                                                          inference_image, 
                                                                                          images_pred_bboxes, 
                                                                                          config["datasets"][dataset_name]["classes"],
                                                                                          log,
                                                                                          draw_bbox=True)

    log.info("Completed inference...", task="inference")

    log.info("Removing temp png image...", task="inference")
    os.remove(inference_image_png)

    return inference_image_png, inference_image_bbox, inference_bboxes