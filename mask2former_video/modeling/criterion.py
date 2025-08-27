# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# ------------------------------------------------------------------------------------------------
# Modified by Kaixuan Lu from https://github.com/facebookresearch/CutLER/tree/main/videocutler
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized
from einops import rearrange


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def dice_loss_weight(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    loss_weight: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss * loss_weight
    return loss.sum() / num_masks


dice_loss_weight_jit = torch.jit.script(dice_loss_weight)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss_weight(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    loss_weight: torch.Tensor,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1)
    loss = loss * loss_weight

    return loss.sum() / num_masks


sigmoid_ce_loss_weight_jit = torch.jit.script(sigmoid_ce_loss_weight)  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, mask_the_feature=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.mask_the_feature = mask_the_feature

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_labels_drop(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs

        def get_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = target_masks.shape
            pred_masks = pred_masks.unsqueeze(1)
            # resize pred_masks to match target_masks
            pred_masks = F.interpolate(
                pred_masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.sigmoid()
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks.detach()
            target_masks = target_masks > 0.5
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(pred_masks, target_masks)
            union = torch.logical_or(pred_masks, target_masks)
            frame_ious = intersection.sum(1) / (union.sum(1) + 1e-6)
            return frame_ious

        def get_best_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = pred_masks.shape
            best_ious = [0.0] * t
            best_ious = torch.tensor(best_ious, device=pred_masks.device)
            best_mean_ious = -1e-5
            for target_mask in target_masks:
                frame_ious = get_frame_ious(pred_masks, target_mask)
                mean_ious = frame_ious.mean()
                if mean_ious > best_mean_ious:
                    best_mean_ious = mean_ious
                    best_ious = frame_ious
            return best_ious

        target_masks = [target["masks"] for target in targets]
        total_frame_ious = []
        for pred_masks, target_mask in zip(outputs["pred_masks"], target_masks):
            frame_ious = []
            for pred_mask in pred_masks:
                frame_ious.append(get_best_frame_ious(pred_mask, target_mask))
            frame_ious = torch.stack(frame_ious, dim=0)
            total_frame_ious.append(frame_ious)
        total_frame_ious = torch.stack(total_frame_ious, dim=0)
        total_frame_ious = total_frame_ious.mean(dim=2)
        ious_weight = total_frame_ious.le(0.01).float()
        ious_weight = 1 - ious_weight.ge(1.0).float()
        # for iou, weight in zip(
        #     total_frame_ious.flatten(0).tolist(),
        #     ious_weight.flatten(0).tolist(),
        # ):
        #     print(f"Frame iou: {iou:.3f}, weight: {weight:.3f}")

        src_logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            reduction="none",
        )
        # loss_ce = loss_ce * ious_weight
        loss_ce = loss_ce * ious_weight.detach()
        if ious_weight.sum() == 0:
            loss_ce = loss_ce.sum()
        else:
            loss_ce = loss_ce.sum() / ious_weight.sum()
        losses = {"loss_ce": loss_ce}
        
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_drop(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        # Modified to handle video
        target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)]).to(
            src_masks
        )

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        def get_point_ious(
            point_logits: torch.Tensor,
            point_labels: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute the point-wise IOU between the predicted and target masks.
            Args:
                point_logits: A float tensor of shape (N, 1, H, W) containing the predicted masks.
                point_labels: A float tensor of shape (N, 1, H, W) containing the target masks.
            Returns:
                A float tensor of shape (N,) containing the IOU for each mask.
            """
            point_logits = point_logits.sigmoid()
            intersection = (point_logits * point_labels).sum(-1)
            union = (point_logits + point_labels).sum(-1) - intersection
            iou = intersection / union
            return iou
        point_ious = get_point_ious(point_logits, point_labels)
        loss_weight = point_ious.le(0.01).float()
        loss_weight = 1 - loss_weight.ge(1.0).float()

        loss_weight = loss_weight.detach()

        losses = {
            "loss_mask": sigmoid_ce_loss_weight_jit(point_logits, point_labels, num_masks, loss_weight),
            "loss_dice": dice_loss_weight_jit(point_logits, point_labels, num_masks, loss_weight),
        }

        del src_masks
        del target_masks
        return losses
    
    def loss_maskiou(self, outputs, targets, indices, num_masks):
        """
        Compute the loss for the predicted IoU.
        Args:
            outputs (dict): The outputs of the model.
            targets (list): The targets of the model.
            indices (list): The indices of the matching between the outputs and the targets.
            num_masks (int): The number of masks in the batch.
        Returns:
            loss (dict): The loss for the predicted IoU.
        """
        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)]).to(
            src_masks
        )

        def get_video_ious(
            outputs: torch.Tensor,
            targets: torch.Tensor,
        ):
            b, t, _, _ = outputs.shape
            src_masks = outputs.flatten(0, 1)[:, None]
            target_masks = targets.flatten(0, 1)[:, None]
            # resize the src_masks to the same shape as target_masks
            src_masks = torch.nn.functional.interpolate(
                src_masks,
                size=target_masks.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            src_masks = src_masks.sigmoid()
            src_masks = src_masks > 0.5
            target_masks = target_masks > 0.5
            src_masks = src_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(src_masks, target_masks)
            union = torch.logical_or(src_masks, target_masks)
            ious = torch.sum(intersection.float(), dim=1) / (
                torch.sum(union.float(), dim=1) + 1e-6
            )

            ious = rearrange(ious, "(b t) -> b t ()", b=b, t=t)
            ious = ious.clamp(0, 1)

            return ious

        if len(src_masks) == 0:
            return {
                "loss_maskiou": torch.tensor(0.0, device=src_masks.device),
            }
        ious = get_video_ious(
            src_masks,
            target_masks,
        )
        ious = ious.detach()

        pred_maskiou = outputs["pred_maskiou"]
        pred_maskiou = pred_maskiou[src_idx]

        # use l2 loss
        loss = F.mse_loss(pred_maskiou, ious, reduction="none")
        loss = loss.sum(dim=1)
        loss = loss.sum() / num_masks

        del src_masks
        del target_masks
        del ious
        del pred_maskiou
        return {
            "loss_maskiou": loss,
        }
    
    def loss_maskiou_drop(self, outputs, targets, indices, num_masks):
        """
        Compute the loss for the predicted IoU.
        Args:
            outputs (dict): The outputs of the model.
            targets (list): The targets of the model.
            indices (list): The indices of the matching between the outputs and the targets.
            num_masks (int): The number of masks in the batch.
        Returns:
            loss (dict): The loss for the predicted IoU.
        """
        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)]).to(
            src_masks
        )

        def get_video_ious(
            outputs: torch.Tensor,
            targets: torch.Tensor,
        ):
            b, t, _, _ = outputs.shape
            src_masks = outputs.flatten(0, 1)[:, None]
            target_masks = targets.flatten(0, 1)[:, None]
            # resize the src_masks to the same shape as target_masks
            src_masks = torch.nn.functional.interpolate(
                src_masks,
                size=target_masks.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            src_masks = src_masks.sigmoid()
            src_masks = src_masks > 0.5
            target_masks = target_masks > 0.5
            src_masks = src_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(src_masks, target_masks)
            union = torch.logical_or(src_masks, target_masks)
            ious = torch.sum(intersection.float(), dim=1) / (
                torch.sum(union.float(), dim=1) + 1e-6
            )

            ious = rearrange(ious, "(b t) -> b t ()", b=b, t=t)
            ious = ious.clamp(0, 1)

            return ious

        if len(src_masks) == 0:
            return {
                "loss_maskiou": torch.tensor(0.0, device=src_masks.device),
            }
        ious = get_video_ious(
            src_masks,
            target_masks,
        )
        ious = ious.detach()

        pred_maskiou = outputs["pred_maskiou"]
        pred_maskiou = pred_maskiou[src_idx]

        # use l2 loss
        loss = F.mse_loss(pred_maskiou, ious, reduction="none")
        ious_weight = ious.le(0.01).float()
        ious_weight = 1 - ious_weight.ge(1.0).float()
        loss = loss * ious_weight
        loss = loss.sum(dim=1)
        loss = loss.sum() / num_masks

        del src_masks
        del target_masks
        del ious
        del pred_maskiou
        return {
            "loss_maskiou": loss,
        }
    
    def loss_maskiou_all(self, outputs, targets, indices, num_masks):
        """
        Compute the loss for the predicted IoU.
        Args:
            outputs (dict): The outputs of the model.
            targets (list): The targets of the model.
            indices (list): The indices of the matching between the outputs and the targets.
            num_masks (int): The number of masks in the batch.
        Returns:
            loss (dict): The loss for the predicted IoU.
        """
        target_masks = [target["masks"] for target in targets]

        def get_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = target_masks.shape
            pred_masks = pred_masks.unsqueeze(1)
            # resize pred_masks to match target_masks
            pred_masks = F.interpolate(
                pred_masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.sigmoid()
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks.detach()
            target_masks = target_masks > 0.5
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(pred_masks, target_masks)
            union = torch.logical_or(pred_masks, target_masks)
            frame_ious = intersection.sum(1) / (union.sum(1) + 1e-6)
            return frame_ious

        def get_best_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = pred_masks.shape
            best_ious = [0.0] * t
            best_ious = torch.tensor(best_ious, device=pred_masks.device)
            best_mean_ious = -1e-5
            for target_mask in target_masks:
                frame_ious = get_frame_ious(pred_masks, target_mask)
                mean_ious = frame_ious.mean()
                if mean_ious > best_mean_ious:
                    best_mean_ious = mean_ious
                    best_ious = frame_ious
            return best_ious

        total_frame_ious = []
        for pred_masks, target_mask in zip(outputs["pred_masks"], target_masks):
            frame_ious = []
            for pred_mask in pred_masks:
                frame_ious.append(get_best_frame_ious(pred_mask, target_mask))
            frame_ious = torch.stack(frame_ious, dim=0)
            total_frame_ious.append(frame_ious)
        total_frame_ious = torch.stack(total_frame_ious, dim=0)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print(total_frame_ious.detach().cpu().numpy())
        pred_maskiou = outputs["pred_maskiou"].squeeze(3)
        pred_maskiou = pred_maskiou.flatten(0)
        total_frame_ious = total_frame_ious.flatten(0)
        loss = F.mse_loss(
            pred_maskiou,
            total_frame_ious,
            reduction="mean",
        )

        return {
            "loss_maskiou": loss,
        }
    
    def loss_maskiou_all_drop(self, outputs, targets, indices, num_masks):
        """
        Compute the loss for the predicted IoU.
        Args:
            outputs (dict): The outputs of the model.
            targets (list): The targets of the model.
            indices (list): The indices of the matching between the outputs and the targets.
            num_masks (int): The number of masks in the batch.
        Returns:
            loss (dict): The loss for the predicted IoU.
        """
        target_masks = [target["masks"] for target in targets]
        def get_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = target_masks.shape
            pred_masks = pred_masks.unsqueeze(1)
            # resize pred_masks to match target_masks
            pred_masks = F.interpolate(
                pred_masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.sigmoid()
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks.detach()
            target_masks = target_masks > 0.5
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(pred_masks, target_masks)
            union = torch.logical_or(pred_masks, target_masks)
            frame_ious = intersection.sum(1) / (union.sum(1) + 1e-6)
            return frame_ious
        def get_best_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = pred_masks.shape
            best_ious = [0.0] * t
            best_ious = torch.tensor(best_ious, device=pred_masks.device)
            best_mean_ious = -1e-5
            for target_mask in target_masks:
                frame_ious = get_frame_ious(pred_masks, target_mask)
                mean_ious = frame_ious.mean()
                if mean_ious > best_mean_ious:
                    best_mean_ious = mean_ious
                    best_ious = frame_ious
            return best_ious
        total_frame_ious = []
        for pred_masks, target_mask in zip(outputs["pred_masks"], target_masks):
            frame_ious = []
            for pred_mask in pred_masks:
                frame_ious.append(get_best_frame_ious(pred_mask, target_mask))
            frame_ious = torch.stack(frame_ious, dim=0)
            total_frame_ious.append(frame_ious)
        total_frame_ious = torch.stack(total_frame_ious, dim=0)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print(total_frame_ious.detach().cpu().numpy())
        pred_maskiou = outputs["pred_maskiou"].squeeze(3)
        pred_maskiou = pred_maskiou.flatten(0)
        total_frame_ious = total_frame_ious.flatten(0)
        loss = F.mse_loss(
            pred_maskiou,
            total_frame_ious,
            reduction="none",
        )
        ious_weight = total_frame_ious.le(0.01).float()
        ious_weight = 1 - ious_weight.ge(1.0).float()
        loss = loss * ious_weight
        loss = torch.mean(loss)


        return {
            "loss_maskiou": loss,
        }
    
    def loss_maskiou_bal(self, outputs, targets, indices, num_masks):
        """
        Compute the loss for the predicted IoU.
        Args:
            outputs (dict): The outputs of the model.
            targets (list): The targets of the model.
            indices (list): The indices of the matching between the outputs and the targets.
            num_masks (int): The number of masks in the batch.
        Returns:
            loss (dict): The loss for the predicted IoU.
        """
        target_masks = [target["masks"] for target in targets]

        def get_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = target_masks.shape
            pred_masks = pred_masks.unsqueeze(1)
            # resize pred_masks to match target_masks
            pred_masks = F.interpolate(
                pred_masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.sigmoid()
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks.detach()
            target_masks = target_masks > 0.5
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(pred_masks, target_masks)
            union = torch.logical_or(pred_masks, target_masks)
            frame_ious = intersection.sum(1) / (union.sum(1) + 1e-6)
            return frame_ious

        def get_best_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = pred_masks.shape
            best_ious = [0.0] * t
            best_ious = torch.tensor(best_ious, device=pred_masks.device)
            best_mean_ious = -1e-5
            for target_mask in target_masks:
                frame_ious = get_frame_ious(pred_masks, target_mask)
                mean_ious = frame_ious.mean()
                if mean_ious > best_mean_ious:
                    best_mean_ious = mean_ious
                    best_ious = frame_ious
            return best_ious

        total_frame_ious = []
        for pred_masks, target_mask in zip(outputs["pred_masks"], target_masks):
            frame_ious = []
            for pred_mask in pred_masks:
                frame_ious.append(get_best_frame_ious(pred_mask, target_mask))
            frame_ious = torch.stack(frame_ious, dim=0)
            total_frame_ious.append(frame_ious)
        total_frame_ious = torch.stack(total_frame_ious, dim=0)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print(total_frame_ious.detach().cpu().numpy())
        pred_maskiou = outputs["pred_maskiou"].squeeze(3)
        pred_maskiou = pred_maskiou.flatten(0)
        total_frame_ious = total_frame_ious.flatten(0)

        def equally_sample(
            total_frame_ious: torch.Tensor,
            interval_num: int = 10,
            sample_num: int = 10,
        ):
            """
            Equally sample the total_frame_ious tensor.
            Return the sample indices.
            Each interval has sample_num samples.
            """
            min_iou = 0.0
            max_iou = 1.0
            interval = (max_iou - min_iou) / interval_num
            sample_indices = []
            for i in range(interval_num):
                start = min_iou + i * interval
                end = min_iou + (i + 1) * interval
                indices = torch.where(
                    (total_frame_ious >= start) & (total_frame_ious < end)
                )[0]
                if len(indices) > sample_num:
                    # randomly sample sample_num indices
                    # randomly shuffle the indices
                    indices = indices[torch.randperm(len(indices))]
                    indices = indices[:sample_num]

                # print(
                #     f"Interval {i}: {start:.2f} - {end:.2f}, sample num: {len(indices)}"
                # )
                # print(total_frame_ious[indices].tolist())
                sample_indices.append(indices)
            sample_indices = torch.cat(sample_indices, dim=0)
            return sample_indices

        sample_indices = equally_sample(total_frame_ious)
        pred_maskiou = pred_maskiou[sample_indices]
        total_frame_ious = total_frame_ious[sample_indices]
        loss = F.mse_loss(
            pred_maskiou,
            total_frame_ious,
            reduction="mean",
        )

        return {
            "loss_maskiou": loss,
        }

    def loss_vit_maskiou(
        self,
        outputs,
        targets,
        indices,
        num_masks,
    ):
        def get_batch_indices(preds_class, score_threshold=0.8):
            preds_score = F.softmax(preds_class, dim=-1)[:, :, :-1]
            preds_score = preds_score.flatten(1, 2)
            query_indices = preds_score >= score_threshold
            query_indices = query_indices.nonzero(as_tuple=True)
            index_dict = {}
            for id1, id2 in zip(*query_indices):
                id1 = id1.item()
                id2 = id2.item()
                if id1 not in index_dict:
                    index_dict[id1] = []
                index_dict[id1].append(id2)
            batch_indices = []
            for id1, indices in index_dict.items():
                batch_indices.append(([id1] * len(indices), indices))
            return batch_indices

        batch_indices = get_batch_indices(
            outputs["pred_logits"], score_threshold=0.8
        )
        pred_masks = outputs["pred_masks"]
        batch_preds_masks = []
        for batch_index in batch_indices:
            batch_preds_masks.append(pred_masks[batch_index])

        target_masks = [target["masks"] for target in targets]

        def get_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = target_masks.shape
            pred_masks = pred_masks.unsqueeze(1)
            # resize pred_masks to match target_masks
            pred_masks = F.interpolate(
                pred_masks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks = pred_masks.squeeze(1)
            pred_masks = pred_masks.sigmoid()
            pred_masks = pred_masks > 0.5
            pred_masks = pred_masks.detach()
            target_masks = target_masks > 0.5
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)
            intersection = torch.logical_and(pred_masks, target_masks)
            union = torch.logical_or(pred_masks, target_masks)
            frame_ious = intersection.sum(1) / (
                union.sum(1) + 1e-6
            )
            return frame_ious

        def get_best_frame_ious(
            pred_masks,
            target_masks,
        ):
            t, h, w = pred_masks.shape
            best_ious = [0.0] * t
            best_ious = torch.tensor(best_ious, device=pred_masks.device)
            best_mean_ious = -1e-5
            for target_mask in target_masks:
                frame_ious = get_frame_ious(pred_masks, target_mask)
                mean_ious = frame_ious.mean()
                if mean_ious > best_mean_ious:
                    best_mean_ious = mean_ious
                    best_ious = frame_ious
            return best_ious

        total_frame_ious = []
        for pred_masks, target_mask in zip(batch_preds_masks, target_masks):
            frame_ious = []
            for pred_mask in pred_masks:
                frame_ious.append(get_best_frame_ious(pred_mask, target_mask))
            frame_ious = torch.stack(frame_ious, dim=0)
            frame_ious = frame_ious.mean(dim=0)
            total_frame_ious.append(frame_ious)
        if len(total_frame_ious) == 0:
            return {
                "loss_vit_maskiou": torch.tensor(0.0, device=pred_masks.device),
            }
        total_frame_ious = torch.stack(total_frame_ious, dim=0)
        pred_frame_ious = outputs["pred_frame_ious"]
        if pred_frame_ious is None:
            return {
                "loss_vit_maskiou": torch.tensor(0.0, device=pred_masks.device),
            }
        loss = F.mse_loss(
            pred_frame_ious,
            total_frame_ious,
            reduction="mean",
        )
        return {
            "loss_vit_maskiou": loss,
        }
    
    

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'labels_drop': self.loss_labels_drop,
            'masks': self.loss_masks,
            'masks_drop': self.loss_masks_drop,
            'maskiou': self.loss_maskiou,
            'maskiou_drop': self.loss_maskiou_drop,
            'maskiou_all': self.loss_maskiou_all,
            'maskiou_all_drop': self.loss_maskiou_all_drop,
            'maskiou_bal': self.loss_maskiou_bal,
            'vit_maskiou': self.loss_vit_maskiou,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if self.mask_the_feature and loss == "maskiou":
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
