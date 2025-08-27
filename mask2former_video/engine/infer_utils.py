import os
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask
import glob
import gc
import logging
import copy
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
from tqdm import tqdm
import torch
from typing import Any


def get_one_frame_average_iou(one_frame_results):
    ious = [result["iou"] for result in one_frame_results["results"]]
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)


def get_median_iou(results):
    ious = []
    for frame_results in results:
        ious.append(get_one_frame_average_iou(frame_results))
    return float(np.median(ious))


def save_one_frame_results(one_frame_results, base_path):
    file_name = one_frame_results["file_name"]
    raw_anno_file = file_name.replace("JPEGImages", "Annotations").replace(
        ".jpg", ".png"
    )
    raw_anno = Image.open(raw_anno_file)
    anno_palette = raw_anno.getpalette()
    raw_anno.close()
    image = Image.open(file_name)
    video_name = file_name.split("/")[-2]
    image_name = file_name.split("/")[-1].replace(".jpg", "")
    img_dir = os.path.join(base_path, "JPEGImages", video_name)
    anno_dir = os.path.join(base_path, "Annotations", video_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    anno = np.zeros(image.size[::-1], dtype=np.uint8)
    for result in one_frame_results["results"]:
        mask = result["best_mask"]
        object_id = result["object_id"] + 1
        mask = cv2.resize(
            mask.astype(np.int8), image.size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        anno[mask] = object_id
    anno = Image.fromarray(anno)
    anno.putpalette(anno_palette)
    anno.save(os.path.join(anno_dir, f"{image_name}.png"))
    image.save(os.path.join(img_dir, f"{image_name}.jpg"))
    image.close()
    anno.close()


def save_results_with_iou_threshold(results, base_path, iou_threshold):
    for frame_results in results:
        if get_one_frame_average_iou(frame_results) > iou_threshold:
            save_one_frame_results(frame_results, base_path)
    gc.collect()


def create_info():
    return ""


def get_file_names(base_path, video_name):
    """Return file names in the directory.
    Args:
        base_path (str): Base path.
        video_name (str): Video name.
    """
    base_path += "/JPEGImages"
    file_names = []
    for file_name in glob.glob(f"{base_path}/{video_name}/*.jpg"):
        # Remove the base path and the first slash.
        file_name = file_name[len(base_path) + 1 :]
        file_names.append(file_name)
    # Sort the file names.
    file_names.sort()
    return file_names


def get_video_names(base_path):
    """Return video names in the directory.
    Args:
        base_path (str): Base path.
    """
    base_path += "/JPEGImages"
    video_names = []
    for video_name in glob.glob(f"{base_path}/*"):
        # only get the video that has more than 1 frame
        if len(glob.glob(f"{video_name}/*.jpg")) < 2:
            continue
        # Remove the base path.
        video_name = video_name[len(base_path) + 1 :]
        video_names.append(video_name)
    # Sort the video names.
    video_names.sort()
    return video_names


def get_image_size(base_path, video_name):
    """Return image size.
    Args:
        base_path (str): Base path.
        video_name (str): Video name.
    """
    image_path = glob.glob(f"{base_path}/JPEGImages/{video_name}/*.jpg")[0]
    image = Image.open(image_path)
    return image.size


def create_video_info(
    video_id,
    file_names,
    image_size,
    date_captured="2019-04-11 00:18:45.652544",
    license_id=1,
    coco_url="",
    flickr_url="",
):
    """Return video_info in YouTubeVIS format.
    Args:
        video_id (str): Video ID.
        file_names (list): List of file names.
        image_size (tuple): Image size.
        date_captured (str): Date captured.
        license_id (int): License ID.
        coco_url (str): COCO URL.
        flickr_url (str): Flickr URL.
    """
    video_info = {
        "id": video_id,
        "width": image_size[0],
        "height": image_size[1],
        "length": len(file_names),
        "file_names": file_names,
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }
    return video_info


def create_all_video_infos(base_path):
    """Return all video_infos in YouTubeVIS format.
    Args:
        base_path (str): Base path.
    """
    video_infos = []
    video_names = get_video_names(base_path)
    for video_id, video_name in enumerate(video_names):
        file_names = get_file_names(base_path, video_name)
        image_size = get_image_size(base_path, video_name)
        video_info = create_video_info(video_id, file_names, image_size)
        video_infos.append(video_info)
    return video_infos


def get_anno_values(base_path, video_name):
    """Return annotation values.
    Args:
        base_path (str): Base path.
        video_name (str): Video name.
    """
    annos = []
    # get the first annotation image
    # if base_path == "val_half":
    #     base_path = "val"
    #     video_name = video_name.replace("_1", "")
    #     video_name = video_name.replace("_2", "")
    # elif base_path == "train_half":
    #     base_path = "train"
    #     video_name = video_name.replace("_1", "")
    #     video_name = video_name.replace("_2", "")
    if "half" in base_path:
        base_path = base_path.replace("_half", "")
        video_name = video_name.replace("_1", "")
        video_name = video_name.replace("_2", "")
    if "inf" in base_path:
        base_path = "/work3/s232248/final/CutLER/videocutler/datasets/DAVIS/train"
        video_name = video_name.replace("_1", "")
        video_name = video_name.replace("_2", "")

    anno_path = glob.glob(f"{base_path}/Annotations/{video_name}/*.png")[0]
    anno = Image.open(anno_path)
    anno = np.array(anno)
    # get unique values
    annos = np.unique(anno)
    annos.sort()
    annos = annos[1:]
    return annos


def get_one_anno(raw_anno: np.array, anno_value: int):
    """Return mask with one anno.
    Args:
        raw_anno (np.array): Raw annotation.
        anno_id (int): Annotation ID.
    """
    anno = np.zeros_like(raw_anno)
    anno[raw_anno == anno_value] = 1
    anno = anno.astype(np.uint8)
    return anno


def get_one_anno_info(anno: np.array):
    """Return annotation info.
    Args:
        anno (np.array): Annotation.
    """
    binary_mask_encoded = mask.encode(np.asfortranarray(anno.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return {
            "segmentation": None,
            "area": None,
            "bbox": None,
        }

    bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask.encode(np.array(anno[..., None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("ascii")
    segmentation = rle
    return {
        "segmentation": segmentation,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
    }


def create_annotation_info(
    base_path,
    anno_id,
    video_id,
    anno_value,
    video_name,
):
    """Return annotation info.
    Args:
        base_path (str): Base path.
        anno_id (int): Annotation ID.
        video_id (int): Video ID.
        anno_value (int): Annotation value.
        video_name (str): Video name.
    """
    anno_paths = glob.glob(f"{base_path}/Annotations/{video_name}/*.png")
    anno_paths.sort()
    one_anno_infos = []
    for anno_path in anno_paths:
        anno = Image.open(anno_path)
        anno = np.array(anno)
        one_anno = get_one_anno(anno, anno_value)
        one_anno_info = get_one_anno_info(one_anno)
        one_anno_infos.append(one_anno_info)
    segmentations = [one_anno_info["segmentation"] for one_anno_info in one_anno_infos]
    areas = [one_anno_info["area"] for one_anno_info in one_anno_infos]
    bboxes = [one_anno_info["bbox"] for one_anno_info in one_anno_infos]
    annotation_info = {
        "id": anno_id,
        "video_id": video_id,
        "category_id": 1,
        "segmentations": segmentations,
        "areas": areas,
        "bboxes": bboxes,
        "iscrowd": 0,
    }
    return annotation_info


def create_all_anno_infos(base_path):
    """Return all annotation infos.
    Args:
        base_path (str): Base path.
    """
    anno_infos = []
    video_names = get_video_names(base_path)
    anno_id = 0
    for video_id, video_name in enumerate(video_names):
        anno_values = get_anno_values(base_path, video_name)
        for anno_value in anno_values:
            anno_info = create_annotation_info(
                base_path,
                anno_id,
                video_id,
                anno_value,
                video_name,
            )
            anno_id += 1
            anno_infos.append(anno_info)
    return anno_infos


def create_categories():
    return [
        {
            "supercategory": "object",
            "id": 1,
            "name": "object",
        },
    ]


def create_total_info(base_path):
    """Return total info.
    Args:
        base_path (str): Base path.
    """
    info = create_info()
    video_infos = create_all_video_infos(base_path)
    anno_infos = create_all_anno_infos(base_path)
    categories = create_categories()
    return {
        "info": info,
        "videos": video_infos,
        "annotations": anno_infos,
        "categories": categories,
    }


def count_frame_num(total_info):
    """Return frame number.
    Args:
        total_info (dict): Total info.
    """
    frame_num = 0
    for video_info in total_info["videos"]:
        frame_num += video_info["length"]
    return frame_num


def count_video_num(total_info):
    """Return video number.
    Args:
        total_info (dict): Total info.
    """
    video_num = len(total_info["videos"])
    return video_num


def restruct_results(results):
    results_dict = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in results_dict:
            results_dict[video_id] = {}
        anno_infos = result["anno_infos"]
        ious = result["ious"]
        file_names = result["file_names"]
        height = result["height"]
        width = result["width"]
        object_id = result["object_id"]
        for frame_idx in range(len(anno_infos)):
            frame_result = {
                "video_id": video_id,
                "file_name": file_names[frame_idx],
                "height": height,
                "width": width,
                "object_id": object_id,
                "anno_info": anno_infos[frame_idx],
                "iou": ious[frame_idx],
            }
            if frame_idx not in results_dict[video_id]:
                results_dict[video_id][frame_idx] = []
            results_dict[video_id][frame_idx].append(frame_result)
    return results_dict


def filter_results_dict(results_dict, iou, min_frame_num=2):
    filtered_results = {}
    for video_id, frames_dict in results_dict.items():
        filtered_frames = []
        for frame_idx, frame_results in frames_dict.items():
            frame_ious = []
            for frame_result in frame_results:
                frame_ious.append(frame_result["iou"])
            mean_frame_iou = np.mean(frame_ious)
            if mean_frame_iou >= iou:
                filtered_frames.append(frame_results)
        if len(filtered_frames) >= min_frame_num:
            filtered_results[video_id] = filtered_frames
    return filtered_results


def build_video_infos(filtered_results):
    video_infos = []
    for video_id, frame_dicts in filtered_results.items():
        height = frame_dicts[0][0]["height"]
        width = frame_dicts[0][0]["width"]
        file_names = []
        for frame_dict in frame_dicts:
            file_names.append(frame_dict[0]["file_name"])
        video_info = create_video_info(
            video_id=video_id,
            file_names=file_names,
            image_size=(width, height),
        )
        video_infos.append(video_info)
    return video_infos


def build_anno_infos(filtered_results):
    anno_infos = []
    anno_id = 1
    for video_id, frame_dicts in filtered_results.items():
        for object_id in range(len(frame_dicts[0])):
            segmentations = []
            areas = []
            bboxes = []
            for frame_dict in frame_dicts:
                if object_id >= len(frame_dict):
                    # print(
                    #     f"object_id {object_id} is out of range in frame_dict {frame_dict}")
                    # get the logger
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"object_id {object_id} is out of range in frame_dict {frame_dict}"
                    )
                frame_info = frame_dict[object_id]
                segmentations.append(frame_info["anno_info"]["segmentation"])
                areas.append(frame_info["anno_info"]["area"])
                bboxes.append(frame_info["anno_info"]["bbox"])
            anno_infos.append(
                {
                    "id": anno_id,
                    "video_id": video_id,
                    "category_id": 1,
                    "segmentations": segmentations,
                    "areas": areas,
                    "bboxes": bboxes,
                    "iscrowd": 0,
                }
            )
            anno_id += 1
    return anno_infos


def build_total_info(filtered_results):
    info = create_info()
    video_infos = build_video_infos(filtered_results)
    anno_infos = build_anno_infos(filtered_results)
    categories = create_categories()
    return {
        "info": info,
        "videos": video_infos,
        "annotations": anno_infos,
        "categories": categories,
    }


def build_total_info_from_raw_res(
    results, iou_threshold, min_frame_num=2, output_raw=False
):
    results_dict = restruct_results(results)
    filtered_results = filter_results_dict(results_dict, iou_threshold, min_frame_num)
    total_info = build_total_info(filtered_results)
    if output_raw:
        return total_info, filtered_results
    else:
        return total_info


def combine_frame_info(old_frame_info, new_frame_info):
    combined_frame_info = []
    if len(old_frame_info) != len(new_frame_info):
        # print(
        #     f"old_frame_info and new_frame_info have different length: {len(old_frame_info)} and {len(new_frame_info)}"
        # )
        # print(
        #     f"old_frame_info: {old_frame_info} and new_frame_info: {new_frame_info}"
        # )
        logger = logging.getLogger(__name__)
        logger.warning(
            f"old_frame_info and new_frame_info have different length: {len(old_frame_info)} and {len(new_frame_info)}"
        )
        # print the file names
        old_file_names = [info["file_name"] for info in old_frame_info]
        new_file_names = [info["file_name"] for info in new_frame_info]
        logger.warning(f"old_file_names: {old_file_names}")
        logger.warning(f"new_file_names: {new_file_names}")
        # logger.warning(
        #     f"old_frame_info: {old_frame_info} and new_frame_info: {new_frame_info}"
        # )
        # return the longer one
        if len(old_frame_info) > len(new_frame_info):
            return old_frame_info
        else:
            return new_frame_info
    for old_info, new_info in zip(old_frame_info, new_frame_info):
        old_iou = old_info["iou"]
        new_iou = new_info["iou"]
        if old_iou > new_iou:
            combined_frame_info.append(old_info)
        else:
            combined_frame_info.append(new_info)
    return combined_frame_info


def combine_video_info(old_video_info, new_video_info):
    video_info_dict = {}
    old_frame_info_length = len(old_video_info[0])
    new_frame_info_length = len(new_video_info[0])
    if old_frame_info_length != new_frame_info_length:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"old_video_info and new_video_info have different length: {old_frame_info_length} and {new_frame_info_length}"
        )
        logger.warning(
            f"old_video_info: {old_video_info} and new_video_info: {new_video_info}"
        )
        # return the longer one
        if old_frame_info_length > new_frame_info_length:
            return old_video_info
        else:
            return new_video_info
    for frame_info in old_video_info:
        file_name = frame_info[0]["file_name"]
        video_info_dict[file_name] = frame_info
    for frame_info in new_video_info:
        file_name = frame_info[0]["file_name"]
        if file_name not in video_info_dict:
            video_info_dict[file_name] = frame_info
        else:
            old_frame_info = video_info_dict[file_name]
            combined_frame_info = combine_frame_info(old_frame_info, frame_info)
            video_info_dict[file_name] = combined_frame_info
    combined_video_info = list(video_info_dict.values())
    # sort the combined_video_info by file_name
    combined_video_info.sort(key=lambda x: x[0]["file_name"])
    return combined_video_info


def combine_raw_info(old_raw_info, new_raw_info):
    combined_raw_info = {}
    for video_id in old_raw_info.keys():
        if video_id in new_raw_info.keys():
            combined_video_info = combine_video_info(
                old_raw_info[video_id], new_raw_info[video_id]
            )
            combined_raw_info[video_id] = combined_video_info
        else:
            combined_raw_info[video_id] = old_raw_info[video_id]
    for video_id in new_raw_info.keys():
        if video_id not in combined_raw_info.keys():
            combined_raw_info[video_id] = new_raw_info[video_id]
    # sort the combined_raw_info by video_id
    combined_raw_info = dict(sorted(combined_raw_info.items()))
    return combined_raw_info


def restruct_signal_results(results):
    results_dict = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in results_dict:
            results_dict[video_id] = {}
        anno_infos = result["anno_infos"]
        mask_signals = result["mask_signals"]
        file_names = result["file_names"]
        height = result["height"]
        width = result["width"]
        mean_mask_signal = result["mean_mask_signal"]
        for frame_idx in range(len(anno_infos)):
            frame_result = {
                "video_id": video_id,
                "file_name": file_names[frame_idx],
                "height": height,
                "width": width,
                "mean_mask_signal": mean_mask_signal,
                "mask_signal": mask_signals[frame_idx],
                "anno_info": anno_infos[frame_idx],
            }
            if frame_idx not in results_dict[video_id]:
                results_dict[video_id][frame_idx] = []
            results_dict[video_id][frame_idx].append(frame_result)
    return results_dict


def filter_signal_dict(results_dict, signal_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frames_dict in results_dict.items():
        filtered_frames = []
        for frame_idx, frame_results in frames_dict.items():
            frame_signals = []
            for frame_result in frame_results:
                frame_signals.append(frame_result["mask_signal"])
            mean_frame_signal = np.mean(frame_signals)
            if mean_frame_signal <= signal_threshold:
                filtered_frames.append(frame_results)
        if len(filtered_frames) >= min_frame_num:
            filtered_results[video_id] = filtered_frames
    return filtered_results


def get_video_average_signal(video_info):
    video_signals = []
    for frame_info in video_info[0]:
        video_signals.append(frame_info["mean_mask_signal"])
    video_signals = np.array(video_signals)
    return np.mean(video_signals)


def combine_raw_info_with_signal(old_raw_info, new_raw_info):
    combined_raw_info = copy.deepcopy(old_raw_info)
    for video_id, new_video_info in new_raw_info.items():
        if video_id not in combined_raw_info:
            combined_raw_info[video_id] = new_video_info
        else:
            new_signal = get_video_average_signal(new_video_info)
            old_signal = get_video_average_signal(combined_raw_info[video_id])
            if new_signal < old_signal:
                combined_raw_info[video_id] = new_video_info
    # sort the combined raw info by video id
    combined_raw_info = dict(sorted(combined_raw_info.items(), key=lambda x: int(x[0])))

    return combined_raw_info


def restruct_score_results(results):
    results_dict = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in results_dict:
            results_dict[video_id] = {}
        anno_infos = result["anno_infos"]
        mask_signals = result["mask_signals"]
        file_names = result["file_names"]
        height = result["height"]
        width = result["width"]
        score = result["score"]
        mean_mask_signal = result["mean_mask_signal"]
        for frame_idx in range(len(anno_infos)):
            frame_result = {
                "video_id": video_id,
                "file_name": file_names[frame_idx],
                "height": height,
                "width": width,
                "mean_mask_signal": mean_mask_signal,
                "mask_signal": mask_signals[frame_idx],
                "anno_info": anno_infos[frame_idx],
                "score": score,
            }
            if frame_idx not in results_dict[video_id]:
                results_dict[video_id][frame_idx] = []
            results_dict[video_id][frame_idx].append(frame_result)
    return results_dict


def fileter_score_dict(results_dict, score_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frame_dict in results_dict.items():
        filtered_frames = []
        for frame_idx, frame_results in frame_dict.items():
            frame_scores = []
            for frame_result in frame_results:
                frame_scores.append(frame_result["score"])
            mean_frame_score = np.mean(frame_scores)
            if mean_frame_score >= score_threshold:
                filtered_frames.append(frame_results)
        if len(filtered_frames) >= min_frame_num:
            filtered_results[video_id] = filtered_frames
    return filtered_results


def restruct_prediou_results(results):
    results_dict = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in results_dict:
            results_dict[video_id] = {}
        anno_infos = result["anno_infos"]
        file_names = result["file_names"]
        height = result["height"]
        width = result["width"]
        pred_ious = result["pred_ious"]
        for frame_idx in range(len(anno_infos)):
            frame_result = {
                "video_id": video_id,
                "file_name": file_names[frame_idx],
                "height": height,
                "width": width,
                "anno_info": anno_infos[frame_idx],
                "pred_iou": pred_ious[frame_idx],
            }
            if frame_idx not in results_dict[video_id]:
                results_dict[video_id][frame_idx] = []
            results_dict[video_id][frame_idx].append(frame_result)
    return results_dict


def filter_prediou_dict(results_dict, pred_iou_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frame_dict in results_dict.items():
        filterer_frames = []
        for frame_idx, frame_results in frame_dict.items():
            frame_predious = []
            for frame_result in frame_results:
                frame_predious.append(frame_result["pred_iou"])
            mean_frame_predious = np.mean(frame_predious)
            if mean_frame_predious >= pred_iou_threshold:
                filterer_frames.append(frame_results)
        if len(filterer_frames) >= min_frame_num:
            filtered_results[video_id] = filterer_frames
    return filtered_results


def get_perc_prediou_threshold(
    results_dict,
    percentage,
):
    frame_pred_ious = []
    for frame_dict in results_dict.values():
        for frame_results in frame_dict.values():
            frame_predious = []
            for frame_result in frame_results:
                frame_predious.append(frame_result["pred_iou"])
            mean_predious = np.mean(frame_predious)
            frame_pred_ious.append(mean_predious)
    frame_pred_ious = np.array(frame_pred_ious)
    frame_pred_ious = np.sort(frame_pred_ious)[::-1]
    if percentage > 1.0 or percentage < 0.0:
        raise ValueError("percentage should be in [0, 1]")
    index = int(len(frame_pred_ious) * percentage)
    return frame_pred_ious[index]


# def restruct_key_results(results, metric_key):
#     results_dict = {}
#     for result in results:
#         video_id = result["video_id"]
#         if video_id not in results_dict:
#             results_dict[video_id] = {}
#         anno_infos = result["anno_infos"]
#         file_names = result["file_names"]
#         height = result["height"]
#         width = result["width"]
#         metrics = result[metric_key]
#         for frame_idx in range(len(anno_infos)):
#             frame_result = {
#                 "video_id": video_id,
#                 "file_name": file_names[frame_idx],
#                 "height": height,
#                 "width": width,
#                 "anno_info": anno_infos[frame_idx],
#                 "metric": metrics[frame_idx],
#             }
#             if frame_idx not in results_dict[video_id]:
#                 results_dict[video_id][frame_idx] = []
#             results_dict[video_id][frame_idx].append(frame_result)
#     return results_dict


def restruct_key_results(results, metric_key):
    results_dict = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in results_dict:
            results_dict[video_id] = {}
        anno_infos = result["anno_infos"]
        file_names = result["file_names"]
        height = result["height"]
        width = result["width"]
        metrics = result[metric_key]
        for frame_idx in range(len(anno_infos)):
            frame_result = {
                "video_id": video_id,
                "file_name": file_names[frame_idx],
                "height": height,
                "width": width,
                "anno_info": anno_infos[frame_idx],
                "metric": metrics[frame_idx],
            }
            if frame_idx not in results_dict[video_id]:
                results_dict[video_id][frame_idx] = []
            results_dict[video_id][frame_idx].append(frame_result)
    return results_dict


def get_perc_metric_threshold(
    results_dict,
    percentage,
):
    frame_metrics = []
    for frame_dict in results_dict.values():
        for frame_results in frame_dict.values():
            tmp_frame_metrics = []
            for frame_result in frame_results:
                tmp_frame_metrics.append(frame_result["metric"])
            mean_metrics = np.mean(tmp_frame_metrics)
            frame_metrics.append(mean_metrics)
    frame_metrics = np.array(frame_metrics)
    frame_metrics = np.sort(frame_metrics)[::-1]
    if percentage > 1.0 or percentage < 0.0:
        raise ValueError("percentage should be in [0, 1]")
    index = int(len(frame_metrics) * percentage)
    return frame_metrics[index]


def get_perc_metric_threshold_min(
    results_dict,
    percentage,
):
    frame_metrics = []
    for frame_dict in results_dict.values():
        for frame_results in frame_dict.values():
            tmp_frame_metrics = []
            for frame_result in frame_results:
                tmp_frame_metrics.append(frame_result["metric"])
            min_metrics = np.min(tmp_frame_metrics)
            frame_metrics.append(min_metrics)
    frame_metrics = np.array(frame_metrics)
    frame_metrics = np.sort(frame_metrics)[::-1]
    if percentage > 1.0 or percentage < 0.0:
        raise ValueError("percentage should be in [0, 1]")
    index = int(len(frame_metrics) * percentage)
    return frame_metrics[index]


def get_perc_metric_thresholdV2_min(
    results_dict,
    percentage,
):
    frame_metrics = []
    for frame_dict in results_dict.values():
        for frame_results in frame_dict.values():
            tmp_frame_metrics = []
            for frame_result in frame_results:
                area = frame_result["anno_info"]["area"]
                if area is None:
                    continue
                tmp_frame_metrics.append(frame_result["metric"])
            if len(tmp_frame_metrics) == 0:
                min_metrics = 0.0
            else:
                min_metrics = np.min(tmp_frame_metrics)
            frame_metrics.append(min_metrics)
    frame_metrics = np.array(frame_metrics)
    frame_metrics = np.sort(frame_metrics)[::-1]
    if percentage > 1.0 or percentage < 0.0:
        raise ValueError("percentage should be in [0, 1]")
    index = int(len(frame_metrics) * percentage)
    return frame_metrics[index]


def filter_metric_dict(results_dict, pred_iou_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frame_dict in results_dict.items():
        filterer_frames = []
        for frame_idx, frame_results in frame_dict.items():
            frame_metrics = []
            for frame_result in frame_results:
                frame_metrics.append(frame_result["metric"])
            mean_frame_metrics = np.mean(frame_metrics)
            if mean_frame_metrics >= pred_iou_threshold:
                filterer_frames.append(frame_results)
        if len(filterer_frames) >= min_frame_num:
            filtered_results[video_id] = filterer_frames
    return filtered_results


def filter_metric_dict_min(results_dict, pred_iou_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frame_dict in results_dict.items():
        filterer_frames = []
        for frame_idx, frame_results in frame_dict.items():
            frame_metrics = []
            for frame_result in frame_results:
                frame_metrics.append(frame_result["metric"])
            min_frame_metrics = np.min(frame_metrics)
            if min_frame_metrics >= pred_iou_threshold:
                filterer_frames.append(frame_results)
        if len(filterer_frames) >= min_frame_num:
            filtered_results[video_id] = filterer_frames
    return filtered_results


def filter_metric_dictV2_min(results_dict, pred_iou_threshold, min_frame_num=2):
    filtered_results = {}
    for video_id, frame_dict in results_dict.items():
        filterer_frames = []
        for frame_idx, frame_results in frame_dict.items():
            frame_metrics = []
            for frame_result in frame_results:
                area = frame_result["anno_info"]["area"]
                if area is None:
                    continue
                frame_metrics.append(frame_result["metric"])
            if len(frame_metrics) == 0:
                min_frame_metrics = 0.0
            else:
                min_frame_metrics = np.min(frame_metrics)
            if min_frame_metrics >= pred_iou_threshold:
                filterer_frames.append(frame_results)
        if len(filterer_frames) >= min_frame_num:
            filtered_results[video_id] = filterer_frames
    return filtered_results


class MyYTVIS:
    def __init__(self, data: dict, base_path: str):
        data = copy.deepcopy(data)
        self.base_path = base_path
        self.videos = data["videos"]
        if "JPEGImages" in self.videos[0]["file_names"][0]:

            def get_relative_path(file_name):
                file_name = file_name.split("/")[-2:]
                file_name = "/".join(file_name)
                return file_name

            for video in self.videos:
                video["file_names"] = [
                    get_relative_path(file_name) for file_name in video["file_names"]
                ]
        self.annotations = data["annotations"]
        raw_data = {}
        for video in self.videos:
            video_id = video["id"]
            raw_data[video_id] = {
                "file_names": video["file_names"],
                "annotations": [],
            }
        for annotation in self.annotations:
            video_id = annotation["video_id"]
            raw_data[video_id]["annotations"].append(annotation)
        self.raw_data = raw_data

        video_data = {}
        for video_id, video in self.raw_data.items():
            file_names = video["file_names"]
            annotations = video["annotations"]
            video_name = file_names[0].split("/")[0]
            if video_name not in video_data:
                video_data[video_name] = {}

            for frame_id, file_name in enumerate(file_names):
                frame_annotations = []
                for annotation in annotations:
                    frame_anno = annotation["segmentations"][frame_id]
                    if frame_anno:
                        frame_annotations.append(frame_anno)
                video_data[video_name][file_name] = frame_annotations
        self.video_data = video_data
        self.video_names = list(self.video_data.keys())

    @classmethod
    def decode_anno(cls, anno):
        """
        Decode the mask annotation
        :param anno: The mask annotation
        :return: The decoded mask
        """
        height, width = anno["size"]
        counts = anno["counts"]
        if isinstance(counts, list):
            rle = maskUtils.frPyObjects(anno, height, width)
            mask = maskUtils.decode(rle).astype(bool)
        else:
            mask = maskUtils.decode(anno).astype(bool)
        return mask

    def show_frame(self, video_name: str, frame_name: str, title=None):
        annotations = self.video_data[video_name][frame_name]
        annotations = [self.decode_anno(anno) for anno in annotations]
        object_num = len(annotations)

        image = Image.open(f"{self.base_path}/{frame_name}")
        image = np.array(image)
        if title is None:
            title = f"Object Num: {object_num}"
        else:
            title = f"{title}, Object Num: {object_num}"
        plt.title(title)
        plt.imshow(image)
        # plt.show()

        # Use a colormap with more distinct colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(annotations)))
        for anno, color in zip(annotations, colors):
            plt.imshow(
                np.ma.masked_where(anno == 0, anno),
                cmap=plt.cm.colors.ListedColormap([color]),
                alpha=0.6,
            )
            # plt.show()
        plt.axis("off")
        plt.show()


def count_inter_annos(
    annos_1: list, annos_2: list, iou_threshold: float = 0.5, use_nms: bool = False
):
    """
    Count the number of intersection annotations
    :param annos_1: The first annotation
    :param annos_2: The second annotation
    :param iou_threshold: The IoU threshold
    :return: The number of intersection annotations
    """
    annos_1 = [anno for anno in annos_1 if anno]
    annos_2 = [anno for anno in annos_2 if anno]

    def nms(masks: list, iou_threshold: float = 0.5):
        """
        Non-maximum suppression for masks
        :param masks: The masks to be suppressed
        :param iou_threshold: The IoU threshold
        :return: The suppressed masks
        """
        if len(masks) == 0:
            return []

        keep = []
        while len(masks) > 0:
            current_mask = masks[0]
            keep.append(current_mask)
            masks = masks[1:]

            # Calculate IoU with the current mask
            ious = []
            for mask in masks:
                # use torch cuda to calculate IoU
                current_mask = torch.tensor(current_mask).cuda().bool()
                mask = torch.tensor(mask).cuda().bool()
                intersection = torch.logical_and(current_mask, mask)
                union = torch.logical_or(current_mask, mask)
                iou = torch.sum(intersection) / (torch.sum(union) + 1e-6)
                ious.append(iou.item())

                # intersection = np.logical_and(current_mask, mask)
                # union = np.logical_or(current_mask, mask)
                # iou = np.sum(intersection) / (np.sum(union) + 1e-6)
                # ious.append(iou)

            # Keep only the masks with IoU less than the threshold
            masks = [mask for mask, iou in zip(masks, ious) if iou < iou_threshold]

        return keep

    if len(annos_1) == 0 or len(annos_2) == 0:
        return 0

    masks_1 = [MyYTVIS.decode_anno(anno) for anno in annos_1]
    masks_2 = [MyYTVIS.decode_anno(anno) for anno in annos_2]
    if use_nms:
        masks_1 = nms(masks_1, iou_threshold)
        masks_2 = nms(masks_2, iou_threshold)

    intersection_count = 0
    for mask_1 in masks_1:
        for mask_2 in masks_2:
            intersection = np.logical_and(mask_1, mask_2)
            union = np.logical_or(mask_1, mask_2)
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
            if iou > iou_threshold:
                intersection_count += 1
                break
    return intersection_count


def analysis_yvis(
    gt_yvis: MyYTVIS,
    inf_ytvis: MyYTVIS,
    iou_threshold: float = 0.5,
    use_tqdm: bool = False,
    use_nms: bool = False,
):
    """
    Analysis the YVIS dataset
    :param gt_yvis: The ground truth YVIS dataset
    :param inf_ytvis: The inference YVIS dataset
    :param iou_threshold: The IoU threshold
    :return: The analysis result
    """
    inf_video_names = inf_ytvis.video_names

    gt_object_counts = []
    inf_object_counts = []
    intersection_counts = []
    if use_tqdm:
        inf_video_names = tqdm(inf_video_names)
    for inf_video_name in inf_video_names:
        inf_frame_names = inf_ytvis.video_data[inf_video_name].keys()
        for inf_frame_name in inf_frame_names:
            inf_frame_annos = inf_ytvis.video_data[inf_video_name][inf_frame_name]
            gt_frame_annos = gt_yvis.video_data[inf_video_name][inf_frame_name]
            intersection_count = count_inter_annos(
                inf_frame_annos, gt_frame_annos, iou_threshold, use_nms=use_nms
            )
            intersection_counts.append(intersection_count)
            gt_object_counts.append(len(gt_frame_annos))
            inf_object_counts.append(len(inf_frame_annos))
    gt_object_counts = np.array(gt_object_counts)
    inf_object_counts = np.array(inf_object_counts)
    intersection_counts = np.array(intersection_counts)
    gt_object_counts = gt_object_counts.sum()
    inf_object_counts = inf_object_counts.sum()
    intersection_counts = intersection_counts.sum()
    return {
        "gt_object_counts": gt_object_counts,
        "inf_object_counts": inf_object_counts,
        "intersection_counts": intersection_counts,
    }


def decode_segmentation(anno):
    """
    Decode the mask annotation
    :param anno: The mask annotation
    :return: The decoded mask
    """
    height, width = anno["size"]
    counts = anno["counts"]
    if isinstance(counts, list):
        rle = maskUtils.frPyObjects(anno, height, width)
        mask = maskUtils.decode(rle).astype(bool)
    else:
        mask = maskUtils.decode(anno).astype(bool)
    return mask


def get_object_perc_metric_threshold(
    results_dict: dict[str, Any],
    percentage: float,
):
    object_metrics = []
    for video_results in results_dict.values():
        for frame_results in video_results.values():
            for object_result in frame_results:
                area = object_result["anno_info"]["area"]
                if area is not None:
                    object_metrics.append(object_result["metric"])
    object_metrics = np.array(object_metrics)
    object_metrics = np.sort(object_metrics)[::-1]  # sort in descending order
    index = int(len(object_metrics) * percentage)
    return object_metrics[index].tolist()


def mark_results(results_dict, metric_threshold, min_frame_num=2):
    marked_results_dict = copy.deepcopy(results_dict)
    for video_id, video_results in marked_results_dict.items():
        for frame_id, frame_results in video_results.items():
            for object_id, object_result in enumerate(frame_results):
                area = object_result["anno_info"]["area"]
                if area is None:
                    frame_results[object_id]["marked"] = True
                else:
                    metric = object_result["metric"]
                    if metric < metric_threshold:
                        frame_results[object_id]["marked"] = False
                    else:
                        frame_results[object_id]["marked"] = True

    return marked_results_dict

def visualize_marked_video(
    marked_video_results: dict[str, Any],
):
    frame_results = marked_video_results
    frame_num = len(frame_results)
    img_per_line = 5
    line_num = (frame_num + img_per_line - 1) // img_per_line
    plt.figure(figsize=(img_per_line * 5, line_num *3))
    for frame_id, object_results in frame_results.items():
        plt.subplot(line_num, img_per_line, int(frame_id) + 1)
        object_num = len(object_results)
        file_name = object_results[0]["file_name"]
        img = Image.open(file_name)
        plt.imshow(img)
        plt.axis("off")
        colors = plt.cm.tab20(np.linspace(0, 1, object_num))
        frame_marked = True
        for object_idx, object_result in enumerate(object_results):
            anno_info = object_result["anno_info"]
            if anno_info["segmentation"] is None:
                continue
            mask = decode_segmentation(anno_info["segmentation"])
            color = plt.cm.colors.ListedColormap(colors[object_idx % len(colors)])
            if object_result["marked"]:
                mark_color = "red"
            else:
                frame_marked = False
                mark_color = "blue"
            plt.contour(mask, colors=mark_color, linewidths=1)
            plt.imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap=color,
                alpha=0.5,
            )
            plt.text(
                anno_info["bbox"][0],
                anno_info["bbox"][1],
                f"{object_result['metric']:.2f}",
                color=mark_color,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
            )
        if frame_marked:
            plt.title(f"Frame {frame_id} selected, object num: {object_num}", color="red")
        else:
            plt.title(f"Frame {frame_id} not selected, objcet num: {object_num}", color="blue")
    plt.tight_layout()
    plt.show()

def construct_object_based_dict(marked_results_dict):
    object_based_dict = {}
    for video_id, video_results in marked_results_dict.items():
        new_video_results = {}
        for frame_id, frame_results in video_results.items():
            for object_id, object_result in enumerate(frame_results):
                if object_id not in new_video_results:
                    new_video_results[object_id] = {}
                new_video_results[object_id][frame_id] = object_result
        object_based_dict[video_id] = new_video_results
    return object_based_dict

def construct_frame_based_dict(object_based_dict):
    frame_based_dict = {}
    for video_id, video_results in object_based_dict.items():
        new_video_results = {}
        for object_id, object_results in video_results.items():
            for frame_id, frame_result in object_results.items():
                if frame_id not in new_video_results:
                    new_video_results[frame_id] = []
                new_video_results[frame_id].append(frame_result)
        frame_based_dict[video_id] = new_video_results
    return frame_based_dict

def remove_low_quality_objects(
    object_based_results: dict[str, dict[str, Any]], min_frame_num=2
):
    """
    Remove objects that are not marked in enough frames.
    :param object_based_results: The object based results
    :param min_frame_num: The minimum number of frames an object should be marked in
    :return: The filtered object based results
    """
    new_object_based_results = {}
    for video_id, video_results in object_based_results.items():
        new_video_results = {}
        for object_id, object_results in video_results.items():
            marked_frame_num = sum(
                1 for frame_result in object_results.values() if frame_result["marked"] and frame_result["anno_info"]["segmentation"] is not None
            )
            if marked_frame_num >= min_frame_num:
                new_video_results[object_id] = object_results
        if len(new_video_results) > 0:
            new_object_based_results[video_id] = new_video_results
    return new_object_based_results

def merge_object_based_results(
    old_object_based_results: dict[str, dict[str, Any]],
    new_object_based_results: dict[str, dict[str, Any]],
):
    """
    Merge two object based results.
    :param old_object_based_results: The old object based results
    :param new_object_based_results: The new object based results
    :return: The merged object based results
    """
    merged_results = copy.deepcopy(old_object_based_results)

    def check_object_same(
        old_object_results: dict[str, Any],
        new_object_results: dict[str, Any],
        iou_threshold=0.5,
    ):
        """
        Check if two object results are the same.
        :param old_object_results: The old object results
        :param new_object_results: The new object results
        :param iou_threshold: The IoU threshold
        :return: True if the two objects are the same, False otherwise
        """
        for frame_id, old_frame_result in old_object_results.items():
            if frame_id not in new_object_results:
                raise ValueError(
                    f"Frame {frame_id} not found in new object results for old object {old_object_results}"
                )
            new_frame_result = new_object_results[frame_id]
            old_segmentation = old_frame_result["anno_info"]["segmentation"]
            new_segmentation = new_frame_result["anno_info"]["segmentation"]
            if old_segmentation is None or new_segmentation is None:
                continue
            old_mask = decode_segmentation(old_segmentation)
            new_mask = decode_segmentation(new_segmentation)
            union = np.logical_or(old_mask, new_mask)
            if union.sum() < 1e-6:
                continue
            intersection = np.logical_and(old_mask, new_mask)
            iou = intersection.sum() / union.sum()
            if iou >= iou_threshold:
                return True
        return False

    for video_id, new_video_results in new_object_based_results.items():
        if video_id not in merged_results:
            merged_results[video_id] = new_video_results
        else:
            for _, new_object_results in new_video_results.items():
                max_object_id = max(merged_results[video_id].keys(), default=0)
                current_object_id = max_object_id + 1
                merged_flag = False
                for old_object_id, old_object_results in merged_results[
                    video_id
                ].items():
                    if check_object_same(
                        old_object_results, new_object_results
                    ):
                        merged_objects = {}
                        for frame_id, new_frame_result in new_object_results.items():
                            if frame_id not in old_object_results:
                                raise ValueError(
                                    f"Frame {frame_id} not found in old object results for new object {new_object_results}"
                                )
                            old_frame_result = old_object_results[frame_id]
                            new_frame_marked = new_frame_result["marked"]
                            old_frame_marked = old_frame_result["marked"]
                            if new_frame_marked or not old_frame_marked:
                                merged_objects[frame_id] = copy.deepcopy(
                                    new_frame_result
                                )
                            else:
                                merged_objects[frame_id] = copy.deepcopy(
                                    old_frame_result
                                )
                        merged_results[video_id][old_object_id] = merged_objects
                        merged_flag = True
                        break
                if not merged_flag:
                    merged_results[video_id][current_object_id] = copy.deepcopy(
                        new_object_results
                    )
                    current_object_id += 1

    return merged_results

def filter_marked_results(
    marked_results: dict[str, dict[str, Any]],
    min_frame_num=2,
):
    """
    Filter the marked results to only keep the objects that are marked in enough frames.
    :param marked_results: The marked results
    :param min_frame_num: The minimum number of frames an object should be marked in
    :return: The filtered marked results
    """
    filtered_results = {}
    for video_id, video_results in marked_results.items():
        filter_frames = []
        for frame_id, object_results in video_results.items():
            frame_marked = True
            for object_result in object_results:
                if not object_result["marked"]:
                    frame_marked = False
                    break
            if frame_marked:
                filter_frames.append(object_results)
        if len(filter_frames) >= min_frame_num:
            filtered_results[video_id] = filter_frames
    return filtered_results