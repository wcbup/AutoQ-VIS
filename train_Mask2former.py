from mask2former.utils.misc import is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn

from detectron2.engine import launch

import logging
import numpy as np
import torch
import copy
import cv2
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from PIL import Image
from mask2former_video.engine.infer_utils import (
    create_total_info,
    count_frame_num,
    save_results_with_iou_threshold,
    get_one_frame_average_iou,
    get_median_iou,
    get_one_anno_info,
    create_info,
    create_categories,
    create_video_info,
    count_video_num,
    build_total_info_from_raw_res,
    build_total_info,
    restruct_signal_results,
    filter_signal_dict,
    restruct_prediou_results,
    filter_prediou_dict,
    get_perc_prediou_threshold,
    restruct_key_results,
    get_perc_metric_threshold,
    filter_metric_dict,
    MyYTVIS,
    analysis_yvis,
    get_perc_metric_thresholdV2_min,
    filter_metric_dictV2_min,
    get_object_perc_metric_threshold,
    construct_object_based_dict,
    construct_frame_based_dict,
    remove_low_quality_objects,
    merge_object_based_results,
    filter_marked_results,
    mark_results,
)
import json
import pathlib
import os
from mask2former_video.data_video.datasets.ytvis import register_ytvis_instances
from mask2former_video.data_video.datasets.builtin import (
    _get_ytvis_2019_small_instances_meta,
)
from pycocotools import mask
from scipy.stats import pearsonr, spearmanr


def get_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    # # resize the pred mask to the gt mask
    # pred = cv2.resize(pred.astype(np.int8), gt.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

    # resize the gt mask to the pred mask
    gt = cv2.resize(
        gt.astype(np.int8), pred.shape[::-1], interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # use torch tensor to calculate the iou
    gt = torch.from_numpy(gt).cuda()
    pred = torch.from_numpy(pred).cuda()
    intersection = torch.logical_and(gt, pred)
    union = torch.logical_or(gt, pred)
    if union.sum() < 1e-6:
        return 0.0
    iou = intersection.sum() / union.sum()
    return iou.item()

    # intersection = np.logical_and(gt, pred)
    # union = np.logical_or(gt, pred)
    # if union.sum() < 1e-6:
    #     return 0.0
    # iou = np.sum(intersection) / np.sum(union)
    # return iou


# def get_frame_masks(ouputs, frame_id):
#     frame_masks = []
#     for query_masks in ouputs["pred_masks"]:
#         frame_masks.append(query_masks[frame_id].cpu().detach().numpy())
#     return np.array(frame_masks)


# def find_best_mask(gt_mask: np.ndarray, frame_masks: np.ndarray):
#     ious = [get_iou(gt_mask, mask) for mask in frame_masks]
#     best_index = np.argmax(ious)
#     return ious[best_index], frame_masks[best_index]


# def eval_one_frame(inputs, ouputs, frame_id):
#     frame_masks = get_frame_masks(ouputs, frame_id)
#     frame_gt_masks = (
#         inputs[0]["instances"][frame_id].gt_masks.tensor.cpu().detach().numpy()
#     )
#     results = []
#     for object_id in range(len(frame_gt_masks)):
#         iou, best_mask = find_best_mask(frame_gt_masks[object_id], frame_masks)
#         results.append(
#             {
#                 "iou": iou,
#                 "best_mask": best_mask,
#                 "gt_mask": frame_gt_masks[object_id],
#                 "object_id": object_id,
#             }
#         )
#     return results


# def eval_one_video(inputs, ouputs):
#     results = []
#     for frame_id in range(len(inputs[0]["image"])):
#         file_name = inputs[0]["file_names"][frame_id]
#         one_frame_results = eval_one_frame(inputs, ouputs, frame_id)
#         results.append({"file_name": file_name, "results": one_frame_results})
#     return results


# class MyEvaluator(DatasetEvaluator):
#     def __init__(self):
#         super().__init__()
#         self.results = []
#         self.logger = logging.getLogger(__name__)

#     def reset(self):
#         self.results = []

#     def process(self, inputs, outputs):
#         inputs = copy.deepcopy(inputs)
#         outputs = copy.deepcopy(outputs)
#         results = eval_one_video(inputs, outputs)
#         self.results.extend(results)

#     def evaluate(self):
#         iou_list = []
#         for frame_results in self.results:
#             iou_list.extend([result["iou"] for result in frame_results["results"]])
#         mean_iou = np.mean(iou_list)
#         self.logger.info(f"Mean IoU: {mean_iou:.2f}")
#         return copy.deepcopy(
#             {
#                 "mean_iou": mean_iou,
#                 "results": self.results,
#             }
#         )


def get_instance_masks(instances):
    # get the instance video
    instance_video = []
    for i in range(len(instances)):
        instance_video.append(instances[i].gt_masks.tensor.cpu().detach().numpy())
    return np.array(instance_video)


def get_video_ious(gt_video, pred_video):
    ious = []
    for gt_mask, pred_mask in zip(gt_video, pred_video):
        iou = get_iou(gt_mask, pred_mask)
        ious.append(float(iou))
    return ious


def find_best_video_mask(gt_video: np.array, pred_videos: np.array):
    # ious = [get_video_iou(gt_video, pred_video) for pred_video in pred_videos]
    # best_index = np.argmax(ious)
    # anno_infos = [
    #     get_one_anno_info(frame_anno) for frame_anno in pred_videos[best_index]
    # ]
    # return ious[best_index], best_index, anno_infos
    best_index = 0
    best_mean_iou = 0
    anno_infos = []
    best_ious = []
    for query_id, pred_video in enumerate(pred_videos):
        ious = get_video_ious(gt_video, pred_video)
        mean_iou = np.mean(ious)
        if mean_iou >= best_mean_iou:
            best_mean_iou = mean_iou
            best_index = query_id
            best_ious = ious
            anno_infos = [get_one_anno_info(frame_anno) for frame_anno in pred_video]
    return best_mean_iou, best_index, anno_infos, best_ious


# def eval_one_video(inputs, outputs):
#     results = []
#     video_id = inputs[0]["video_id"]
#     file_names = inputs[0]["file_names"]
#     height = inputs[0]["height"]
#     width = inputs[0]["width"]
#     instance_masks = get_instance_masks(inputs[0]["instances"])
#     pred_videos = np.array(outputs["pred_masks"])
#     for object_id in range(instance_masks.shape[1]):
#         object_iou, query_id, anno_infos, ious = find_best_video_mask(
#             instance_masks[:, object_id], pred_videos
#         )
#         object_iou = float(object_iou)
#         query_id = int(query_id)
#         results.append(
#             {
#                 "mean_iou": object_iou,
#                 "query_id": query_id,
#                 "object_id": object_id,
#                 "video_id": video_id,
#                 "file_names": file_names,
#                 "height": height,
#                 "width": width,
#                 "anno_infos": anno_infos,
#                 "ious": ious,
#             }
#         )
#     return results


# class MyYTVISEvaluator(DatasetEvaluator):
#     def __init__(self):
#         super().__init__()
#         self.results = []
#         self.logger = logging.getLogger(__name__)

#     def reset(self):
#         self.results = []
#         self.logger = logging.getLogger(__name__)

#     def process(self, inputs, outputs):
#         inputs = copy.deepcopy(inputs)
#         outputs = copy.deepcopy(outputs)
#         results = eval_one_video(inputs, outputs)
#         self.results.extend(results)

#     def evaluate(self):
#         return copy.deepcopy(
#             {
#                 "results": self.results,
#             }
#         )


def get_mask_signal(one_raw_mask: torch.Tensor):
    one_raw_mask = one_raw_mask[one_raw_mask > 0.01]
    if len(one_raw_mask) == 0:
        return 0.0
    one_raw_mask = one_raw_mask - 0.5
    one_raw_mask = one_raw_mask.abs()
    mask_signal = one_raw_mask
    mask_signal = mask_signal / 0.5
    mask_signal = 1 - mask_signal
    mask_signal = mask_signal.mean().item()
    return mask_signal


def get_best_one_video_ious(pred_video: np.array, instance_masks: np.array):
    best_mean_iou = 0
    best_ious = []
    best_object_id = 0
    for object_id in range(instance_masks.shape[1]):
        ious = get_video_ious(instance_masks[:, object_id], pred_video)
        mean_iou = np.mean(ious)
        if mean_iou >= best_mean_iou:
            best_mean_iou = mean_iou
            best_ious = ious
            best_object_id = object_id
    return best_mean_iou, best_ious, best_object_id


# def eval_one_video_using_signal(
#     inputs,
#     outputs,
#     score_threshold=0.7,
#     output_ious=False,
# ):
#     video_id = inputs[0]["video_id"]
#     file_names = inputs[0]["file_names"]
#     height = inputs[0]["height"]
#     width = inputs[0]["width"]
#     pred_videos = np.array(outputs["pred_masks"])
#     pred_scores = np.array(outputs["pred_scores"])
#     raw_masks = outputs["raw_masks"].sigmoid()
#     mask_signals = []
#     if output_ious:
#         instance_masks = get_instance_masks(inputs[0]["instances"])
#     for video in raw_masks:
#         video_mask_signals = []
#         for one_raw_mask in video:
#             mask_signal = get_mask_signal(one_raw_mask)
#             video_mask_signals.append(mask_signal)
#         mask_signals.append(video_mask_signals)
#     mask_signals = np.array(mask_signals)

#     # select the idx of the video
#     # use the score threshold to select the video
#     results = []
#     for idx, pred_score in enumerate(pred_scores):
#         if pred_score >= score_threshold:
#             pred_video = pred_videos[idx]
#             anno_infos = [get_one_anno_info(frame_anno) for frame_anno in pred_video]
#             mean_signal = np.mean(mask_signals[idx])
#             mean_signal = float(mean_signal)
#             results.append(
#                 {
#                     "mean_mask_signal": mean_signal,
#                     "video_id": video_id,
#                     "file_names": file_names,
#                     "height": height,
#                     "width": width,
#                     "anno_infos": anno_infos,
#                     "mask_signals": mask_signals[idx].tolist(),
#                 }
#             )
#             if output_ious:
#                 mean_iou, ious, object_id = get_best_one_video_ious(pred_video, instance_masks)
#                 mean_iou = float(mean_iou)
#                 results[-1]["mean_iou"] = mean_iou
#                 results[-1]["ious"] = ious
#                 results[-1]["object_id"] = object_id
#     return results


# class MySignalEvaluator(DatasetEvaluator):
#     def __init__(self, output_ious=False):
#         super().__init__()
#         self.results = []
#         self.logger = logging.getLogger(__name__)
#         self.output_ious = output_ious

#     def reset(self):
#         self.results = []
#         self.logger = logging.getLogger(__name__)

#     def process(self, inputs, outputs):
#         inputs = copy.deepcopy(inputs)
#         outputs = copy.deepcopy(outputs)
#         results = eval_one_video_using_signal(inputs, outputs, output_ious=self.output_ious)
#         self.results.extend(results)

#     def evaluate(self):
#         return copy.deepcopy(
#             {
#                 "results": self.results,
#             }
#         )
    
def apply_nms(outputs, iou_threshold=0.5):
    """
    Apply NMS to the outputs
    :param outputs: The outputs of the model
    :param iou_threshold: The IoU threshold
    :return: The outputs after NMS
    """
    image_size = outputs["image_size"]
    pred_scores = outputs["pred_scores"]
    pred_labels = outputs["pred_labels"]
    pred_masks = outputs["pred_masks"]
    raw_masks = outputs["raw_masks"]
    pred_ious = outputs["pred_ious"]
    pred_frame_ious = outputs["pred_frame_ious"]

    def get_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        # use torch tensor to calculate the iou
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        if union.sum() < 1e-6:
            return 0.0
        iou = intersection.sum() / union.sum()
        return iou.item()

        # intersection = np.logical_and(mask1, mask2)
        # union = np.logical_or(mask1, mask2)
        # if union.sum() < 1e-6:
        #     return 0.0
        # iou = intersection.sum() / union.sum()
        # return iou

    keep_indices = []
    current_indices = list(range(len(pred_scores)))
    while len(current_indices) > 0:
        current_mask = pred_masks[current_indices[0]]
        keep_indices.append(current_indices[0])
        # remove the current mask from the list
        current_indices = current_indices[1:]

        # Calculate the IoU of the current mask with all other masks
        ious = []
        for i in current_indices:
            iou = get_mask_iou(current_mask, pred_masks[i])
            ious.append(iou)
        
        # Remove the masks with IoU greater than the threshold
        current_indices = [
            current_indices[i]
            for i in range(len(current_indices))
            if ious[i] < iou_threshold
        ]

    # Create a new dictionary with the filtered masks
    new_image_size = image_size
    new_pred_scores = []
    new_pred_labels = []
    new_pred_masks = []
    new_raw_masks = []
    new_pred_ious = []
    new_pred_frame_ious = pred_frame_ious
    for i in keep_indices:
        new_pred_scores.append(pred_scores[i])
        new_pred_labels.append(pred_labels[i])
        new_pred_masks.append(pred_masks[i])
        if raw_masks is not None:
            new_raw_masks.append(raw_masks[i])
        else:
            new_raw_masks = None
        if len(pred_ious) > 0:
            new_pred_ious.append(pred_ious[i])
        else:
            new_pred_ious = []
    new_outputs = {
        "image_size": new_image_size,
        "pred_scores": new_pred_scores,
        "pred_labels": new_pred_labels,
        "pred_masks": new_pred_masks,
        "raw_masks": new_raw_masks,
        "pred_ious": new_pred_ious,
        "pred_frame_ious": new_pred_frame_ious,
    }
    return new_outputs

def apply_nmsV2(outputs, iou_threshold=0.5):
    """
    Apply NMS to the outputs
    :param outputs: The outputs of the model
    :param iou_threshold: The IoU threshold
    :return: The outputs after NMS
    """
    image_size = outputs["image_size"]
    pred_scores = outputs["pred_scores"]
    pred_labels = outputs["pred_labels"]
    pred_masks = outputs["pred_masks"]
    raw_masks = outputs["raw_masks"]
    pred_ious = outputs["pred_ious"]
    pred_frame_ious = outputs["pred_frame_ious"]

    def get_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
        # use torch tensor to calculate the iou
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
        intersection = torch.logical_and(mask1, mask2)
        union = torch.logical_or(mask1, mask2)
        if union.sum() < 1e-6:
            return 0.0
        iou = intersection.sum() / union.sum()
        return iou.item()

        # intersection = np.logical_and(mask1, mask2)
        # union = np.logical_or(mask1, mask2)
        # if union.sum() < 1e-6:
        #     return 0.0
        # iou = intersection.sum() / union.sum()
        # return iou

    def get_max_frame_iou(video_mask1: torch.Tensor, video_mask2: torch.Tensor) -> float:
        # use torch tensor to calculate the iou
        ious = []
        for frame_mask1, frame_mask2 in zip(video_mask1, video_mask2):
            iou = get_mask_iou(frame_mask1, frame_mask2)
            ious.append(iou)
        return max(ious)

    keep_indices = []
    current_indices = list(range(len(pred_scores)))
    while len(current_indices) > 0:
        current_mask = pred_masks[current_indices[0]]
        keep_indices.append(current_indices[0])
        # remove the current mask from the list
        current_indices = current_indices[1:]

        # Calculate the IoU of the current mask with all other masks
        ious = []
        for i in current_indices:
            # iou = get_mask_iou(current_mask, pred_masks[i])
            iou = get_max_frame_iou(current_mask, pred_masks[i])
            ious.append(iou)
        
        # Remove the masks with IoU greater than the threshold
        current_indices = [
            current_indices[i]
            for i in range(len(current_indices))
            if ious[i] < iou_threshold
        ]

    # Create a new dictionary with the filtered masks
    new_image_size = image_size
    new_pred_scores = []
    new_pred_labels = []
    new_pred_masks = []
    new_raw_masks = []
    new_pred_ious = []
    new_pred_frame_ious = pred_frame_ious
    for i in keep_indices:
        new_pred_scores.append(pred_scores[i])
        new_pred_labels.append(pred_labels[i])
        new_pred_masks.append(pred_masks[i])
        if raw_masks is not None:
            new_raw_masks.append(raw_masks[i])
        else:
            new_raw_masks = None
        if len(pred_ious) > 0:
            new_pred_ious.append(pred_ious[i])
        else:
            new_pred_ious = []
    new_outputs = {
        "image_size": new_image_size,
        "pred_scores": new_pred_scores,
        "pred_labels": new_pred_labels,
        "pred_masks": new_pred_masks,
        "raw_masks": new_raw_masks,
        "pred_ious": new_pred_ious,
        "pred_frame_ious": new_pred_frame_ious,
    }
    return new_outputs

def get_iou_pred_iou(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    # score_threshold=0.7,
):
    # outputs = apply_nms(outputs, iou_threshold=0.5)
    outputs = apply_nmsV2(outputs, iou_threshold=0.5)
    global INFER_SCORE_THRESHOLD
    score_threshold = INFER_SCORE_THRESHOLD
    video_id = inputs[0]["video_id"]
    file_names = inputs[0]["file_names"]
    # convert the file_names (relative path) to absolute paths
    file_names = [pathlib.Path(fn).resolve() for fn in file_names]
    # convert the file_names to strings
    file_names = [str(fn) for fn in file_names]
    height = inputs[0]["height"]
    width = inputs[0]["width"]
    pred_videos = np.array(outputs["pred_masks"])
    pred_ious = np.array(outputs["pred_ious"])
    instance_masks = get_instance_masks(inputs[0]["instances"])
    pred_scores = np.array(outputs["pred_scores"])

    results = []
    for idx, pred_video in enumerate(pred_videos):
        pred_score = pred_scores[idx]
        if pred_score < score_threshold:
            continue
        anno_infos = [get_one_anno_info(frame_anno) for frame_anno in pred_video]
        mean_iou, ious, object_id = get_best_one_video_ious(pred_video, instance_masks)
        # mean_iou = float(mean_iou)
        pred_iou = pred_ious[idx].squeeze(1)
        score_ious = pred_iou * pred_score
        score_ious = score_ious.tolist()
        pred_iou = pred_iou.tolist()
        results.append(
            {
                "video_id": video_id,
                "file_names": file_names,
                "height": height,
                "width": width,
                "anno_infos": anno_infos,
                # "mean_iou": mean_iou,
                "ious": ious,
                "pred_ious": pred_iou,
                # "object_id": object_id,
                "score_ious": score_ious,
            }
        )
    return results

class MaskIoUEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.results = []
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        self.results = []
        self.logger = logging.getLogger(__name__)
    
    def process(self, inputs, outputs):
        inputs = copy.deepcopy(inputs)
        outputs = copy.deepcopy(outputs)
        results = get_iou_pred_iou(inputs, outputs)
        self.results.extend(results)
    
    def evaluate(self):
        # ious = []
        # score_ious = []
        # for video_results in self.results:
        #     ious.extend(video_results["ious"])
        #     score_ious.extend(video_results["score_ious"])
        # ious = np.array(ious)
        # score_ious = np.array(score_ious)
        # if len(ious) < 1 or len(score_ious) < 1:
        #     self.logger.info("No results found.")
        #     return {
        #         "results": {
        #             "pearson": 0.0,
        #             "spearman": 0.0,
        #             "total_results": self.results,
        #         }
        #     }
        # ious = copy.deepcopy(ious)
        # score_ious = copy.deepcopy(score_ious)
        # pearson_score = pearsonr(ious, score_ious)[0]
        # spearman_score = spearmanr(ious, score_ious)[0]
        comm.synchronize()
        results = comm.gather(self.results, dst=0)
        results = list(itertools.chain(*results))
        self.results = copy.deepcopy(results)
        if not comm.is_main_process():
            return {}
        ious = []
        score_ious = []
        for video_results in self.results:
            ious.extend(video_results["ious"])
            score_ious.extend(video_results["score_ious"])
        ious = np.array(ious)
        score_ious = np.array(score_ious)
        if len(ious) < 1 or len(score_ious) < 1:
            self.logger.info("No results found.")
            return {
                "results": {
                    "pearson": 0.0,
                    "spearman": 0.0,
                    "total_results": self.results,
                }
            }
        ious = copy.deepcopy(ious)
        score_ious = copy.deepcopy(score_ious)
        pearson_score = pearsonr(ious, score_ious)[0]
        spearman_score = spearmanr(ious, score_ious)[0]

        return {
            "results": {
                "pearson": pearson_score,
                "spearman": spearman_score,
                "total_results": self.results,
            }
        }



# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import gc
import numpy as np

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    # DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import DatasetCatalog

# MaskFormer
from mask2former import add_maskformer2_config
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_maskformer2_video_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)

# additional settings
from mask2former_video.engine import DefaultTrainer
import wandb
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# setup wandb


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        # if dataset_name == "ytvis_2019_val_new":
        if len(cfg.DATASETS.TEST) > 1 and dataset_name == cfg.DATASETS.TEST[1]:
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                os.makedirs(output_folder, exist_ok=True)

            return YTVISEvaluator(dataset_name, cfg, True, output_folder)
        # elif (
        #     dataset_name == "ytvis_2019_small"
        #     or dataset_name == "ytvis_2019_train_new"
        #     or dataset_name == cfg.MODEL_NAME
        # ):
        else:
            return MaskIoUEvaluator()
            # if hasattr(cfg, "OUTPUT_RAW_MASKS") and cfg.OUTPUT_RAW_MASKS:
            #     if hasattr(cfg, "OUTPUT_IOUS") and cfg.OUTPUT_IOUS:
            #         return MySignalEvaluator(output_ious=True)
            #     else:
            #         return MySignalEvaluator()
            # else:
            #     return MyYTVISEvaluator()

        raise NotImplementedError()

        # return MyEvaluator()

    @classmethod
    def get_train_loader(cls, cfg, dataset_name):
        mapper = YTVISDatasetMapper(cfg, is_train=True)

        dataset_dict = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=(
                cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
            ),
        )
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_train_loader(cls, cfg):
        # dataset_name = cfg.DATASETS.TRAIN[0]
        # mapper = YTVISDatasetMapper(cfg, is_train=True)

        # dataset_dict = get_detection_dataset_dicts(
        #     dataset_name,
        #     filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        #     proposal_files=(
        #         cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
        #     ),
        # )
        def get_train_loader(cfg, dataset_name):
            mapper = YTVISDatasetMapper(cfg, is_train=True)

            dataset_dict = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=(
                    cfg.DATASETS.PROPOSAL_FILES_TRAIN
                    if cfg.MODEL.LOAD_PROPOSALS
                    else None
                ),
            )
            return build_detection_train_loader(
                cfg, mapper=mapper, dataset=dataset_dict
            )

        import random

        class CombinedLoader:
            def __init__(self, loader1, loader2):
                self.loader1 = loader1
                self.loader2 = loader2
                self.loader1_iter = iter(self.loader1)
                self.loader2_iter = iter(self.loader2)

            def __iter__(self):
                self.loader1_iter = iter(self.loader1)
                self.loader2_iter = iter(self.loader2)
                return self

            def __next__(self):
                if random.random() < 0.5:
                    return next(self.loader1_iter)
                else:
                    return next(self.loader2_iter)

        if len(cfg.DATASETS.TRAIN) == 2:
            loader1 = get_train_loader(cfg, cfg.DATASETS.TRAIN[0])
            loader2 = get_train_loader(cfg, cfg.DATASETS.TRAIN[1])
            return CombinedLoader(loader1, loader2)
        else:
            return get_train_loader(cfg, cfg.DATASETS.TRAIN[0])

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        base_lr_multiplier_names = cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES
        base_lr_multiplier = cfg.SOLVER.BASE_LR_MULTIPLIER

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                if module_name in base_lr_multiplier_names:
                    hyperparams["lr"] *= base_lr_multiplier
                    print(" Checked: ", module_name, hyperparams["lr"])
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            # if comm.is_main_process():
            #     assert isinstance(
            #         results_i, dict
            #     ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            #         results_i
            #     )
            #     logger.info(
            #         "Evaluation results for {} in csv format:".format(dataset_name)
            #     )
            #     print_csv_format(results_i)

            # if comm.is_main_process():
            #     iou = results_i["mean_iou"]
            #     logger.info(f"Mean IoU: {iou:.2f}")
            #     if isinstance(iou, torch.Tensor):
            #         iou = iou.item()
            #     if not isinstance(iou, float):
            #         iou = float(iou)
            # print(f"Mean IoU: {iou:.2f}")

            # best_iou = cfg.best_iou
            # if iou > best_iou:
            #     logger.info(f"Best IoU: {iou:.2f}")
            #     print(f"Best IoU: {iou:.2f}")
            #     cfg.defrost()
            #     cfg.best_iou = float(iou)
            #     cfg.freeze()
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(cfg.OUTPUT_DIR, "best_model.pth"),
            #     )

        if len(results) == 1:
            results = list(results.values())[0]

        gc.collect()
        torch.cuda.empty_cache()
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.set_new_allowed(True)
    cfg.merge_from_list(args.opts)
    # NOTE: need to add following lines to detectron2/detectron2/enginee/defaults.py
    # parser.add_argument("--train-dataset", default="", help="training dataset")
    # parser.add_argument("--test-dataset", default="", help="testing dataset")
    # parser.add_argument("--steps", default=0, help="number of steps to train")
    # parser.add_argument("--model_name", default="mask2former", help="model name")
    # NOTE: need to change detectron2/detectron2/data/transforms/transform.py
    # replace all assert img.shape[:2] == (self.h, self.w) to the following lines
    # try:
    #     img.shape[:2] == (self.h, self.w)
    # except:
    #     (self.h, self.w) = (self.w, self.h)
    #     assert img.shape[:2] == (self.h, self.w)
    if args.test_dataset != "":
        cfg.DATASETS.TEST = ((args.test_dataset),)
    if args.train_dataset != "":
        cfg.DATASETS.TRAIN = ((args.train_dataset),)
    if args.steps != 0:
        cfg.SOLVER.STEPS = (int(args.steps),)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="mask2former_video",
    )
    return cfg

INFER_SCORE_THRESHOLD = 0.25
SOCRE_IOU_THRESHOLD = 0.75
num_gpus = 2

def main():
    model_name = "ytvis_s025iouf_allnf_cn075_adding_refresh_g2_b4"
    test_mode = "ytvis_scoreiou_cn_adding_refresh"
    score_iou_threshold = SOCRE_IOU_THRESHOLD
    round_idx = 0
    model_weights = "OUTPUT-DIR/imgnet_iou_pred_all/model_final.pth"
    arg_text = f"--num-gpus {num_gpus} --config-file configs/imagenet_video/{model_name}.yaml  MODEL.WEIGHTS {model_weights} OUTPUT_DIR OUTPUT-DIR/{model_name} MODEL_NAME {model_name}"
    args = default_argument_parser().parse_args(arg_text.split())
    cfg = setup(args)
    logger = logging.getLogger("mask2former_video")

    logger.info("Training with config:\n{}".format(cfg))

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    
    if comm.is_main_process():
        val_results = res[cfg.DATASETS.TEST[1]]
        ap50 = val_results["segm"]["AP50"]
        logger.info("Current AP50: {:.4f}".format(ap50))

        inf_results = res[cfg.DATASETS.TEST[0]]["results"]["total_results"]

        results_dict = restruct_key_results(inf_results, "score_ious")
        spearman_score = res[cfg.DATASETS.TEST[0]]["results"]["spearman"]
        pearson_score = res[cfg.DATASETS.TEST[0]]["results"]["pearson"]
        logger.info(f"Spearman score: {spearman_score:.4f}")
        logger.info(f"Pearson score: {pearson_score:.4f}")
        print(f"Results dict length: {len(results_dict)}")

        metric_threshold = score_iou_threshold
        logger.info(f"score_iou threshold: {metric_threshold:.4f}")

        new_marked_results = mark_results(
            results_dict, metric_threshold, min_frame_num=2
        )
        new_object_based_results = construct_object_based_dict(new_marked_results)
        new_object_based_results = remove_low_quality_objects(
            new_object_based_results, min_frame_num=2
        )

        new_objcet_based_results_path = f"OUTPUT-DIR/{model_name}/object_based_results_{round_idx}.json"
        with open(new_objcet_based_results_path, "w") as f:
            json.dump(new_object_based_results, f, indent=4)

        new_frame_based_results = construct_frame_based_dict(new_object_based_results)
        filtered_results = filter_marked_results(new_frame_based_results)

        total_info = build_total_info(filtered_results)
        frame_num = count_frame_num(total_info)
        video_num = count_video_num(total_info)
        object_num = len(total_info["annotations"])
        logger.info("Frame number: {}".format(frame_num))
        logger.info("Video number: {}".format(video_num))
        logger.info("Object number: {}".format(object_num))

        inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")

        # gt_info = json.load(open("datasets/ytvis_2019/small_.json", "r"))
        # gt_info = json.load(open("datasets/ytvis_2019/valid_30.json", "r"))
        gt_info = json.load(open("datasets/ytvis_2019/train_new.json", "r"))
        gt_ytvis = MyYTVIS(gt_info, "datasets/ytvis_2019/train/JPEGImages")

        analysis_result = analysis_yvis(gt_yvis=gt_ytvis, inf_ytvis=inf_ytvis)
        gt_object_counts = analysis_result["gt_object_counts"]
        inf_object_counts = analysis_result["inf_object_counts"]
        intersection_counts = analysis_result["intersection_counts"]
        logger.info("GT object counts: {}".format(gt_object_counts))
        logger.info("INF object counts: {}".format(inf_object_counts))
        logger.info("Intersection counts: {}".format(intersection_counts))

    info_path = f"datasets/ytvis_2019/{model_name}.json"
    if comm.is_main_process():
        with open(info_path, "w") as f:
            json.dump(total_info, f, indent=4)
    comm.synchronize()

    image_root_path = "datasets/ytvis_2019/train/JPEGImages"
    dataset_json_path = info_path
    def register_my_dataset(name, dataset_json_path, image_root_path):
        register_ytvis_instances(
            name,
            _get_ytvis_2019_small_instances_meta(),
            dataset_json_path,
            image_root_path,
        )

    register_my_dataset(cfg.MODEL_NAME, dataset_json_path, image_root_path)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.model_name = model_name
    trainer.test_mode = test_mode
    trainer.score_iou_threshold = score_iou_threshold
    if comm.is_main_process():
        trainer.gt_ytvis = gt_ytvis
    trainer.round_idx = round_idx + 1
    trainer.train()

if __name__ == "__main__":
    launch(
        main_func=main,
        num_gpus_per_machine=num_gpus,
        num_machines=1,
        dist_url="auto", 
    )