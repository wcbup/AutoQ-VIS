# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------------------------------
# Modified by Kaixuan Lu from https://github.com/facebookresearch/CutLER/tree/main/videocutler

import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom

from einops import rearrange

logger = logging.getLogger(__name__)

# class MaskIoUFeatureExtractor(nn.Module):
#     """
#     MaskIou head feature extractor.
#     """

#     def __init__(self):
#         super(MaskIoUFeatureExtractor, self).__init__()
        
#         input_channels = 257 

#         self.maskiou_fcn1 = nn.Conv2d(input_channels, 256, 3, 1, 1) 
#         self.maskiou_fcn2 = nn.Conv2d(256, 256, 3, 1, 1) 
#         self.maskiou_fcn3 = nn.Conv2d(256, 256, 3, 1, 1) 
#         self.maskiou_fcn4 = nn.Conv2d(256, 256, 3, 2, 1) 
#         self.maskiou_fc1 = nn.Linear(256*7*7, 1024)
#         self.maskiou_fc2 = nn.Linear(1024, 1024)
#         self.maskiou_fc3 = nn.Linear(1024, 1)

#         for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
#             nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
#             nn.init.constant_(l.bias, 0)

#         for l in [self.maskiou_fc1, self.maskiou_fc2]:
#             nn.init.kaiming_uniform_(l.weight, a=1)
#             nn.init.constant_(l.bias, 0)
        
#         nn.init.normal_(self.maskiou_fc3.weight, mean=0, std=0.01)
#         nn.init.constant_(self.maskiou_fc3.bias, 0)


#     def forward(self, x, mask):
#         mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
#         x = torch.cat((x, mask_pool), 1)
#         x = F.relu(self.maskiou_fcn1(x))
#         x = F.relu(self.maskiou_fcn2(x))
#         x = F.relu(self.maskiou_fcn3(x))
#         x = F.relu(self.maskiou_fcn4(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.maskiou_fc1(x))
#         x = F.relu(self.maskiou_fc2(x))
#         x = self.maskiou_fc3(x)
 
#         return x


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        output_raw_masks: bool = False,
        output_maskiou: bool = False,
        output_maskiou_v2: bool = False,
        vit_maskious: bool = False,
        freeze_all: bool = False,
        train_maskiou: bool = False,
        freeze_maskiou: bool = False,
        maskiou_only_imagenet: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.output_raw_masks = output_raw_masks
        # self.maskiou_feature_extractor = MaskIoUFeatureExtractor()
        self.output_maskiou = output_maskiou
        self.output_maskiou_v2 = output_maskiou_v2
        self.vit_maskious = vit_maskious

        if freeze_all:
            for param in self.parameters():
                param.requires_grad = False
        if train_maskiou:
            for param in self.sem_seg_head.predictor.maskiou_feature_extractor.parameters():
                param.requires_grad = True
        if freeze_maskiou:
            for param in self.sem_seg_head.predictor.maskiou_feature_extractor.parameters():
                param.requires_grad = False
        
        self.maskiou_only_imagenet = maskiou_only_imagenet


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        if hasattr(cfg, "OUTPUT_MASKIOUS") and cfg.OUTPUT_MASKIOUS:
            output_maskiou = True
        else:
            output_maskiou = False
        if hasattr(cfg, "OUTPUT_MASKIOUS_V2") and cfg.OUTPUT_MASKIOUS_V2:
            output_maskiou_v2 = True
        else:
            output_maskiou_v2 = False
        if hasattr(cfg, "VIT_MASKIOUS") and cfg.VIT_MASKIOUS:
            vit_maskious = True
        else:
            vit_maskious = False

        if output_maskiou or output_maskiou_v2:
            weight_dict = {
                "loss_ce": class_weight,
                "loss_mask": mask_weight,
                "loss_dice": dice_weight,
                "loss_maskiou": 20.0,
            }
        elif vit_maskious:
            weight_dict = {
                "loss_ce": class_weight,
                "loss_mask": mask_weight,
                "loss_dice": dice_weight,
                "loss_vit_maskiou": 10.0,
            }
        else:
            weight_dict = {
                "loss_ce": class_weight,
                "loss_mask": mask_weight,
                "loss_dice": dice_weight,
            }

        if hasattr(cfg, "USE_DROPLOSS") and cfg.USE_DROPLOSS:
            use_drop_loss = cfg.USE_DROPLOSS
        else:
            use_drop_loss = False

        if hasattr(cfg, "LABEL_DROPLOSS") and cfg.LABEL_DROPLOSS:
            label_drop_loss = cfg.LABEL_DROPLOSS
            # weight_dict["loss_ce"] = class_weight * 10
        else:
            label_drop_loss = False
        

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if output_maskiou or output_maskiou_v2:
            losses = ["labels", "masks", "maskiou"]
        elif vit_maskious:
            losses = ["labels", "masks", "vit_maskiou"]
        else:
            losses = ["labels", "masks"]

        if hasattr(cfg, "FREEZE_ALL") and cfg.FREEZE_ALL:
            freeze_all = cfg.FREEZE_ALL
        else:
            freeze_all = False

        if hasattr(cfg, "TRAIN_MASKIOU") and cfg.TRAIN_MASKIOU:
            train_maskiou = cfg.TRAIN_MASKIOU
        else:
            train_maskiou = False
        
        if hasattr(cfg, "FREEZE_MASKIOU") and cfg.FREEZE_MASKIOU:
            freeze_maskiou = cfg.FREEZE_MASKIOU
        else:
            freeze_maskiou = False
        
        if freeze_all and train_maskiou:
            losses.remove("labels")
            losses.remove("masks")
        
        if freeze_maskiou:
            losses.remove("maskiou")
        
        
        
        if hasattr(cfg, "LOSS_MASKIOU_ALL") and cfg.LOSS_MASKIOU_ALL:
            loss_maskiou_all = cfg.LOSS_MASKIOU_ALL
        else:
            loss_maskiou_all = False
        
        if hasattr(cfg, "LOSS_MASKIOU_BAL") and cfg.LOSS_MASKIOU_BAL:
            loss_maskiou_bal = cfg.LOSS_MASKIOU_BAL
        else:
            loss_maskiou_bal = False
        
        if label_drop_loss:
            # replace "labels" with "labels_drop"
            for i in range(len(losses)):
                if losses[i] == "labels":
                    losses[i] = "labels_drop"
        
        if use_drop_loss:
            # replace "masks" with "masks_drop"
            for i in range(len(losses)):
                if losses[i] == "masks":
                    losses[i] = "masks_drop"
                if losses[i] == "maskiou":
                    losses[i] = "maskiou_drop"
        
        if loss_maskiou_all:
            # replace "maskiou" with "maskiou_all"
            for i in range(len(losses)):
                if losses[i] == "maskiou":
                    losses[i] = "maskiou_all"
                if losses[i] == "maskiou_drop":
                    losses[i] = "maskiou_all_drop"

        if loss_maskiou_bal and loss_maskiou_all:
            raise ValueError("maskiou_all and maskiou_bal cannot be used at the same time")
        
        if loss_maskiou_bal:
            # replace "maskiou" with "maskiou_bal"
            for i in range(len(losses)):
                if losses[i] == "maskiou":
                    losses[i] = "maskiou_bal"
                if losses[i] == "maskiou_drop":
                    losses[i] = "maskiou_bal"

        if hasattr(cfg, "MASK_THE_FEATURE") and cfg.MASK_THE_FEATURE:
            mask_the_feature = True
        else:
            mask_the_feature = False
        
            
        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            mask_the_feature=mask_the_feature,
        )

        if hasattr(cfg, "MASKIOU_ONLY_IMAGENET") and cfg.MASKIOU_ONLY_IMAGENET:
            maskiou_only_imagenet = cfg.MASKIOU_ONLY_IMAGENET
        else:
            maskiou_only_imagenet = False

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "output_maskiou": output_maskiou,
            "output_maskiou_v2": output_maskiou_v2,
            "vit_maskious": vit_maskious,
            "freeze_all": freeze_all,
            "train_maskiou": train_maskiou,
            "freeze_maskiou": freeze_maskiou,
            "maskiou_only_imagenet": maskiou_only_imagenet,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            if self.maskiou_only_imagenet:
                # only train maskiou on imagenet
                filename = batched_inputs[0]["file_names"][0]
                if "imagenet" not in filename:
                    loss_keys = list(losses.keys())
                    for key in loss_keys:
                        if "maskiou" in key:
                            losses.pop(key)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            if self.output_maskiou or self.output_maskiou_v2:
                mask_iou_results = outputs["pred_maskiou"]
                mask_iou_result = mask_iou_results[0]
            else:
                mask_iou_result = None
            
            if self.vit_maskious:
                frame_maskious = outputs["pred_frame_ious"]
                if frame_maskious is not None:
                    frame_maskious = frame_maskious[0]
            else:
                frame_maskious = None

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, mask_iou_result, frame_maskious)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for i, targets_per_video in enumerate(targets):
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(
        self,
        pred_cls,
        pred_masks,
        img_size,
        output_height,
        output_width,
        mask_ious,
        frame_maskious,
    ):
        if frame_maskious is None:
            t = pred_masks.shape[1]
            frame_maskious = torch.Tensor([0.0] * t)

        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            # TODO: make it configurable
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=True)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]

            if self.output_maskiou or self.output_maskiou_v2:
                mask_ious = mask_ious[topk_indices]
                mask_ious = mask_ious.clamp(0, 1)
                out_ious = mask_ious.tolist()
            else:
                out_ious = []


        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ious = []

        raw_masks = pred_masks if self.output_raw_masks else None

        

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "raw_masks": raw_masks,
            "pred_ious": out_ious,
            "pred_frame_ious": frame_maskious.tolist(),
        }

        return video_output
