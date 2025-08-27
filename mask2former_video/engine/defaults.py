# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# ------------------------------------------------------------------------------------------------
# Modified by Kaixuan Lu from https://github.com/facebookresearch/CutLER/tree/main/videocutler

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from detectron2.engine import TrainerBase
from .train_loop import CustomAMPTrainer, CustomSimpleTrainer

from .infer_utils import (
    save_results_with_iou_threshold,
    create_total_info,
    count_frame_num,
    get_median_iou,
    count_video_num,
    build_total_info_from_raw_res,
    combine_raw_info,
    build_total_info,
    restruct_signal_results,
    filter_signal_dict,
    combine_raw_info_with_signal,
    restruct_score_results,
    fileter_score_dict,
    restruct_prediou_results,
    filter_prediou_dict,
    get_perc_prediou_threshold,
    restruct_key_results,
    get_perc_metric_threshold,
    filter_metric_dict,
    MyYTVIS,
    analysis_yvis,
    get_perc_metric_threshold_min,
    filter_metric_dict_min,
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
import gc
import shutil

__all__ = [
    "create_ddp_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
    "DefaultPredictor",
    "DefaultTrainer",
]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--test-dataset", type=str, default="", help="the dataset used for evaluation"
    )
    parser.add_argument(
        "--train-dataset", type=str, default="", help="the dataset used for training"
    )
    parser.add_argument(
        "--wandb-name", type=str, default="", help="the wandb project name"
    )
    parser.add_argument(
        "--no-segm", action="store_true", help="perform evaluation on detection only"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info(
        "Rank of current process: {}. World size: {}".format(
            rank, comm.get_world_size()
        )
    )
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(
                    PathManager.open(args.config_file, "r").read(), args.config_file
                ),
            )
        )

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info(
                "Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml"))
            )
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )


def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = CustomAMPTrainer(model, data_loader, optimizer, cfg=cfg)
        else:
            self._trainer = CustomSimpleTrainer(model, data_loader, optimizer, cfg=cfg)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.best_iou = 0
        self.model_name = None
        self.test_mode = None
        self.iou_threshold = None
        self.signal_threshold = None
        self.score_threshold = None
        self.pred_iou_threshold = None
        self.pred_iou_percents = None
        self.pred_iou_index = 1
        self.gt_ytvis = None
        self.test_datasets = None
        self.round_idx = None
        self.frame_score_iou_percents = None
        self.frame_pred_iou_index = 1
        self.score_iou_threshold = None

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            gc.collect()
            torch.cuda.empty_cache()
            logger = logging.getLogger(__name__)
            # if "mean_iou" in self._last_eval_results:
            if self.test_mode == "only_iou":
                current_iou = self._last_eval_results["mean_iou"]
            # elif self.model_name == None:
            elif self.test_mode == "davis_inf_0.8":
                val_results = self._last_eval_results["davis_val_half"]
                current_iou = val_results["mean_iou"]

                train_results = self._last_eval_results["davis_train_half"]
                iou_threshold = 0.8
                save_bast_path = "datasets/DAVIS/train_inference"
                info_path = "datasets/DAVIS/train_inference.json"
                logger.info(f"Train mean IoU: {train_results['mean_iou']}")
                logger.info(
                    "Saving train results with IoU threshold: {:.2f}".format(
                        iou_threshold
                    )
                )
                save_results_with_iou_threshold(
                    train_results["results"], save_bast_path, iou_threshold
                )

                total_info = create_total_info(save_bast_path)
                frame_num = count_frame_num(total_info)
                logger.info(f"Train frame number: {frame_num}")
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                # reload the dataloader
                new_loader = self.get_train_loader(cfg, "davis_train_inference")
                self._trainer.data_loader.loader1 = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "davis_inf_med":
                val_results = self._last_eval_results["davis_val_half"]
                current_iou = val_results["mean_iou"]

                train_results = self._last_eval_results["davis_train_half"]
                save_base_path = f"datasets/DAVIS/{self.model_name}"
                info_path = f"datasets/DAVIS/{self.model_name}.json"
                # delete the previous results
                if os.path.exists(save_base_path):
                    shutil.rmtree(save_base_path)
                if os.path.exists(info_path):
                    os.remove(info_path)
                iou_threshold = get_median_iou(train_results["results"])
                logger.info(f"Train mean IoU: {train_results['mean_iou']}")
                logger.info(
                    "Saving train results with IoU threshold: {:.4f}".format(
                        iou_threshold
                    )
                )
                save_results_with_iou_threshold(
                    train_results["results"], save_base_path, iou_threshold
                )
                total_info = create_total_info(save_base_path)
                frame_num = count_frame_num(total_info)
                logger.info(f"Train frame number: {frame_num}")
                with open(info_path, "w") as f:
                    json.dump(total_info, f)

                # reload the dataloader
                new_loader = self.get_train_loader(cfg, self.model_name)
                self._trainer.data_loader.loader1 = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
            elif self.test_mode == "ytvis_default":
                if comm.is_main_process():
                    ap50 = self._last_eval_results["segm"]["AP50"]
                    logger.info("Current AP50: {:.4f}".format(ap50))
                    if ap50 > self.best_iou:
                        self.best_iou = ap50
                        # save the best model
                        self.checkpointer.save("model_best")
                        logger.info("Best model saved as model_best.pth")
                return self._last_eval_results
            elif self.test_mode == "ytvis_infer":
                val_results = self._last_eval_results["ytvis_2019_val_new"]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results["ytvis_2019_train_new"]["results"]
                if self.iou_threshold is None:
                    raise ValueError("iou_threshold is None")
                logger.info(
                    "Saving inference results with IoU threshold: {:.2f}".format(
                        self.iou_threshold
                    )
                )
                total_info = build_total_info_from_raw_res(
                    inf_results, self.iou_threshold
                )
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                self._trainer.data_loader.loader1 = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_infer_pure":
                val_results = self._last_eval_results["ytvis_2019_val_new"]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results["ytvis_2019_train_new"]["results"]
                if self.iou_threshold is None:
                    raise ValueError("iou_threshold is None")
                logger.info(
                    "Saving inference results with IoU threshold: {:.2f}".format(
                        self.iou_threshold
                    )
                )
                total_info = build_total_info_from_raw_res(
                    inf_results, self.iou_threshold
                )
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_infer_aug":
                if comm.is_main_process():
                    val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                    ap50 = val_results["segm"]["AP50"]
                    current_iou = ap50
                    logger.info("Current AP50: {:.4f}".format(ap50))
                    if ap50 > self.best_iou:
                        self.best_iou = ap50
                        # save the best model
                        self.checkpointer.save("model_best")
                        logger.info("Best model saved as model_best.pth")

                    inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                    if self.iou_threshold is None:
                        raise ValueError("iou_threshold is None")
                    logger.info(
                        "Saving inference results with IoU threshold: {:.2f}".format(
                            self.iou_threshold
                        )
                    )

                    info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                    raw_info_path = f"datasets/ytvis_2019/{self.model_name}_raw.json"
                    old_raw_info = json.load(open(raw_info_path, "r"))

                    def replace_str_key_with_int_key(old_raw_info):
                        new_raw_info = {}
                        for key, value in old_raw_info.items():
                            new_key = int(key)
                            new_raw_info[new_key] = value
                        return new_raw_info

                    old_raw_info = replace_str_key_with_int_key(old_raw_info)
                    _, new_raw_info = build_total_info_from_raw_res(
                        inf_results, self.iou_threshold, output_raw=True
                    )
                    combined_raw_info = combine_raw_info(old_raw_info, new_raw_info)
                    with open(raw_info_path, "w") as f:
                        json.dump(combined_raw_info, f)
                    total_info = build_total_info(combined_raw_info)
                    frame_num = count_frame_num(total_info)
                    video_num = count_video_num(total_info)
                    logger.info(f"Frame number: {frame_num}")
                    logger.info(f"Video number: {video_num}")
                    with open(info_path, "w") as f:
                        json.dump(total_info, f)
                comm.synchronize()

                # update the dataloader
                if len(cfg.DATASETS.TRAIN) > 1:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_signal":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                if self.signal_threshold is None:
                    raise ValueError("signal_threshold is None")
                logger.info(
                    "Saving inference results with signal threshold: {:.2f}".format(
                        self.signal_threshold
                    )
                )
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                results_dict = restruct_signal_results(inf_results)
                raw_info = filter_signal_dict(results_dict, self.signal_threshold)
                total_info = build_total_info(raw_info)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                with open(info_path, "w") as f:
                    json.dump(total_info, f)

                # update the dataloader
                if len(cfg.DATASETS.TRAIN) > 1:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_signal_aug":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                if self.signal_threshold is None:
                    raise ValueError("signal_threshold is None")
                logger.info(
                    "Saving inference results with signal threshold: {:.2f}".format(
                        self.signal_threshold
                    )
                )
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                raw_info_path = f"datasets/ytvis_2019/{self.model_name}_raw.json"
                old_raw_info = json.load(open(raw_info_path, "r"))

                def replace_str_key_with_int_key(old_raw_info):
                    new_raw_info = {}
                    for key, value in old_raw_info.items():
                        new_key = int(key)
                        new_raw_info[new_key] = value
                    return new_raw_info

                old_raw_info = replace_str_key_with_int_key(old_raw_info)
                results_dict = restruct_signal_results(inf_results)
                new_raw_info = filter_signal_dict(results_dict, self.signal_threshold)
                combined_raw_info = combine_raw_info_with_signal(
                    old_raw_info, new_raw_info
                )
                total_info = build_total_info(combined_raw_info)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                with open(raw_info_path, "w") as f:
                    json.dump(combined_raw_info, f)

                # update the dataloader
                if len(cfg.DATASETS.TRAIN) > 1:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    new_loader = self.get_train_loader(cfg, self.model_name)
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_test":
                val_results = self._last_eval_results["ytvis_2019_val_new"]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results["ytvis_2019_small"]["results"]
                if self.iou_threshold is None:
                    raise ValueError("iou_threshold is None")
                logger.info(
                    "Saving inference results with IoU threshold: {:.2f}".format(
                        self.iou_threshold
                    )
                )
                total_info = build_total_info_from_raw_res(
                    inf_results, self.iou_threshold
                )
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                self._trainer.data_loader.loader1 = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_iou_pred_test":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))

            elif self.test_mode == "ytvis_iou_pred":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))

                inf_results = inf_results["total_results"]
                logger.info(
                    "Saving inference results with pred_iou threshold: {:.2f}".format(
                        self.pred_iou_threshold
                    )
                )
                results_dict = restruct_prediou_results(inf_results)
                filtered_results = filter_prediou_dict(
                    results_dict, self.pred_iou_threshold
                )
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_iou_pred_perc":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))

                inf_results = inf_results["total_results"]
                pred_iou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1

                results_dict = restruct_prediou_results(inf_results)
                pred_iou_threshold = get_perc_prediou_threshold(
                    results_dict, pred_iou_percent
                )
                logger.info(f"Pred iou threshold: {pred_iou_threshold:.4f}")
                filtered_results = filter_prediou_dict(results_dict, pred_iou_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))

                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "ytvis_scoreiou_perc":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dict(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
            
            elif self.test_mode == "ytvis_scoreiou_perc_nms":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dict(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))
            
            elif self.test_mode == "ytvis_scoreiou_perc_nms_min":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_threshold_min(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dict_min(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))

            elif self.test_mode == "ytvis_scoreiou_perc_nms_minV2":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_thresholdV2_min(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dictV2_min(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))
            
            elif self.test_mode == "ytvis_scoreiou_perc_adding":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                if self.pred_iou_index >= len(self.pred_iou_percents):
                    self.pred_iou_index = len(self.pred_iou_percents) - 1
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                
                metric_threshold = get_object_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                new_marked_results = mark_results(
                    results_dict, metric_threshold, min_frame_num=2
                )
                new_object_based_results = construct_object_based_dict(new_marked_results)
                new_object_based_results = remove_low_quality_objects(
                    new_object_based_results, min_frame_num=2
                )

                old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                def load_object_based_results(file_path):
                    """
                    Load the object based results from a json file.
                    :param file_path: The path to the json file
                    :return: The object based results
                    """
                    with open(file_path, "r") as f:
                        object_based_results = json.load(f)
                    # convert the string keys to integers
                    new_object_based_results = {}
                    for video_id, video_results in object_based_results.items():
                        new_video_results = {}
                        for object_id, object_results in video_results.items():
                            new_object_results = {}
                            for frame_id, frame_result in object_results.items():
                                new_object_results[int(frame_id)] = frame_result
                            new_video_results[int(object_id)] = new_object_results
                        new_object_based_results[int(video_id)] = new_video_results
                    return new_object_based_results
                old_object_based_results = load_object_based_results(
                    old_object_based_results_path
                )
                merged_object_based_results = merge_object_based_results(
                    old_object_based_results, new_object_based_results
                )
                merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                self.round_idx += 1
                with open(merged_objcet_based_results_path, "w") as f:
                    json.dump(merged_object_based_results, f, indent=4)
                merged_frame_based_results = construct_frame_based_dict(
                    merged_object_based_results
                )
                filtered_results = filter_marked_results(
                    merged_frame_based_results, min_frame_num=2
                )

                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))
            
            elif self.test_mode == "ytvis_scoreiou_perc_adding_refresh":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                if self.pred_iou_index >= len(self.pred_iou_percents):
                    self.pred_iou_index = len(self.pred_iou_percents) - 1
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                
                metric_threshold = get_object_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                new_marked_results = mark_results(
                    results_dict, metric_threshold, min_frame_num=2
                )
                new_object_based_results = construct_object_based_dict(new_marked_results)
                new_object_based_results = remove_low_quality_objects(
                    new_object_based_results, min_frame_num=2
                )
                
                old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                def load_object_based_results(file_path):
                    """
                    Load the object based results from a json file.
                    :param file_path: The path to the json file
                    :return: The object based results
                    """
                    with open(file_path, "r") as f:
                        object_based_results = json.load(f)
                    # convert the string keys to integers
                    new_object_based_results = {}
                    for video_id, video_results in object_based_results.items():
                        new_video_results = {}
                        for object_id, object_results in video_results.items():
                            new_object_results = {}
                            for frame_id, frame_result in object_results.items():
                                new_object_results[int(frame_id)] = frame_result
                            new_video_results[int(object_id)] = new_object_results
                        new_object_based_results[int(video_id)] = new_video_results
                    return new_object_based_results
                old_object_based_results = load_object_based_results(
                    old_object_based_results_path
                )
                merged_object_based_results = merge_object_based_results(
                    old_object_based_results, new_object_based_results
                )
                merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                self.round_idx += 1
                with open(merged_objcet_based_results_path, "w") as f:
                    json.dump(merged_object_based_results, f, indent=4)
                merged_frame_based_results = construct_frame_based_dict(
                    merged_object_based_results
                )
                filtered_results = filter_marked_results(
                    merged_frame_based_results, min_frame_num=2
                )

                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))

                self.resume_or_load(resume=False)
                logger.info("Model reloaded for next round of training.")
                model = self._trainer.model
                new_optimizer = self.build_optimizer(cfg, model)
                self._trainer.optimizer = new_optimizer
                self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                logger.info("Optimizer and scheduler reloaded for next round of training.")

                # new_model = self.build_model(cfg)
                # new_optimizer = self.build_optimizer(cfg, new_model)
                # new_model = create_ddp_model(new_model, broadcast_buffers=False)
                # self._trainer.model = new_model
                # self._trainer.optimizer = new_optimizer
                # self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                # self.checkpointer = DetectionCheckpointer(
                #     new_model,
                #     cfg.OUTPUT_DIR,
                #     trainer=weakref.proxy(self),
                # )
                # self.resume_or_load(resume=False)
                # logger.info("Model reloaded for next round of training.")
                # logger.info("Optimizer and scheduler reloaded for next round of training.")
            
            elif self.test_mode == "ytvis_scoreiou_cn_adding_refresh":
                if comm.is_main_process():
                    val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                    ap50 = val_results["segm"]["AP50"]
                    current_iou = ap50
                    logger.info("Current AP50: {:.4f}".format(ap50))
                    if ap50 > self.best_iou:
                        self.best_iou = ap50
                        # save the best model
                        self.checkpointer.save("model_best")
                        logger.info("Best model saved as model_best.pth")
                    inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                    pearson_score = inf_results["pearson"]
                    spearman_score = inf_results["spearman"]
                    logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                    logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                    inf_results = inf_results["total_results"]
                    results_dict = restruct_key_results(inf_results, "score_ious")
                    
                    metric_threshold = self.score_iou_threshold
                    logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                    new_marked_results = mark_results(
                        results_dict, metric_threshold, min_frame_num=2
                    )
                    new_object_based_results = construct_object_based_dict(new_marked_results)
                    new_object_based_results = remove_low_quality_objects(
                        new_object_based_results, min_frame_num=2
                    )
                    
                    old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                    def load_object_based_results(file_path):
                        """
                        Load the object based results from a json file.
                        :param file_path: The path to the json file
                        :return: The object based results
                        """
                        with open(file_path, "r") as f:
                            object_based_results = json.load(f)
                        # convert the string keys to integers
                        new_object_based_results = {}
                        for video_id, video_results in object_based_results.items():
                            new_video_results = {}
                            for object_id, object_results in video_results.items():
                                new_object_results = {}
                                for frame_id, frame_result in object_results.items():
                                    new_object_results[int(frame_id)] = frame_result
                                new_video_results[int(object_id)] = new_object_results
                            new_object_based_results[int(video_id)] = new_video_results
                        return new_object_based_results
                    old_object_based_results = load_object_based_results(
                        old_object_based_results_path
                    )
                    merged_object_based_results = merge_object_based_results(
                        old_object_based_results, new_object_based_results
                    )
                    merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                    self.round_idx += 1
                    with open(merged_objcet_based_results_path, "w") as f:
                        json.dump(merged_object_based_results, f, indent=4)
                    merged_frame_based_results = construct_frame_based_dict(
                        merged_object_based_results
                    )
                    filtered_results = filter_marked_results(
                        merged_frame_based_results, min_frame_num=2
                    )

                    total_info = build_total_info(filtered_results)
                    frame_num = count_frame_num(total_info)
                    video_num = count_video_num(total_info)
                    object_num = len(total_info["annotations"])
                    logger.info("Frame number: {}".format(frame_num))
                    logger.info("Video number: {}".format(video_num))
                    logger.info("Object number: {}".format(object_num))
                    info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                    with open(info_path, "w") as f:
                        json.dump(total_info, f)
                    inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                    analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                    gt_object_counts = analysis_result["gt_object_counts"]
                    inf_object_counts = analysis_result["inf_object_counts"]
                    intersection_counts = analysis_result["intersection_counts"]
                    logger.info("GT object counts: {}".format(gt_object_counts))
                    logger.info("INF object counts: {}".format(inf_object_counts))
                    logger.info("Intersection counts: {}".format(intersection_counts))
                comm.synchronize()
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                self.resume_or_load(resume=False)
                logger.info("Model reloaded for next round of training.")
                model = self._trainer.model
                new_optimizer = self.build_optimizer(cfg, model)
                self._trainer.optimizer = new_optimizer
                self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                logger.info("Optimizer and scheduler reloaded for next round of training.")

            elif self.test_mode == "ytvis_scoreiou_cn_adding":
                if comm.is_main_process():
                    val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                    ap50 = val_results["segm"]["AP50"]
                    current_iou = ap50
                    logger.info("Current AP50: {:.4f}".format(ap50))
                    if ap50 > self.best_iou:
                        self.best_iou = ap50
                        # save the best model
                        self.checkpointer.save("model_best")
                        logger.info("Best model saved as model_best.pth")
                    inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                    pearson_score = inf_results["pearson"]
                    spearman_score = inf_results["spearman"]
                    logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                    logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                    inf_results = inf_results["total_results"]
                    results_dict = restruct_key_results(inf_results, "score_ious")
                    
                    metric_threshold = self.score_iou_threshold
                    logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                    new_marked_results = mark_results(
                        results_dict, metric_threshold, min_frame_num=2
                    )
                    new_object_based_results = construct_object_based_dict(new_marked_results)
                    new_object_based_results = remove_low_quality_objects(
                        new_object_based_results, min_frame_num=2
                    )
                    
                    old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                    def load_object_based_results(file_path):
                        """
                        Load the object based results from a json file.
                        :param file_path: The path to the json file
                        :return: The object based results
                        """
                        with open(file_path, "r") as f:
                            object_based_results = json.load(f)
                        # convert the string keys to integers
                        new_object_based_results = {}
                        for video_id, video_results in object_based_results.items():
                            new_video_results = {}
                            for object_id, object_results in video_results.items():
                                new_object_results = {}
                                for frame_id, frame_result in object_results.items():
                                    new_object_results[int(frame_id)] = frame_result
                                new_video_results[int(object_id)] = new_object_results
                            new_object_based_results[int(video_id)] = new_video_results
                        return new_object_based_results
                    old_object_based_results = load_object_based_results(
                        old_object_based_results_path
                    )
                    merged_object_based_results = merge_object_based_results(
                        old_object_based_results, new_object_based_results
                    )
                    merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                    self.round_idx += 1
                    with open(merged_objcet_based_results_path, "w") as f:
                        json.dump(merged_object_based_results, f, indent=4)
                    merged_frame_based_results = construct_frame_based_dict(
                        merged_object_based_results
                    )
                    filtered_results = filter_marked_results(
                        merged_frame_based_results, min_frame_num=2
                    )

                    total_info = build_total_info(filtered_results)
                    frame_num = count_frame_num(total_info)
                    video_num = count_video_num(total_info)
                    object_num = len(total_info["annotations"])
                    logger.info("Frame number: {}".format(frame_num))
                    logger.info("Video number: {}".format(video_num))
                    logger.info("Object number: {}".format(object_num))
                    info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                    with open(info_path, "w") as f:
                        json.dump(total_info, f)
                    inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                    analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                    gt_object_counts = analysis_result["gt_object_counts"]
                    inf_object_counts = analysis_result["inf_object_counts"]
                    intersection_counts = analysis_result["intersection_counts"]
                    logger.info("GT object counts: {}".format(gt_object_counts))
                    logger.info("INF object counts: {}".format(inf_object_counts))
                    logger.info("Intersection counts: {}".format(intersection_counts))
                comm.synchronize()
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                # self.resume_or_load(resume=False)
                # logger.info("Model reloaded for next round of training.")
                # model = self._trainer.model
                # new_optimizer = self.build_optimizer(cfg, model)
                # self._trainer.optimizer = new_optimizer
                # self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                # logger.info("Optimizer and scheduler reloaded for next round of training.")
            
            elif self.test_mode == "ytvis_scoreiou_perc_adding_refreshOp":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                if self.pred_iou_index >= len(self.pred_iou_percents):
                    self.pred_iou_index = len(self.pred_iou_percents) - 1
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                
                metric_threshold = get_object_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                new_marked_results = mark_results(
                    results_dict, metric_threshold, min_frame_num=2
                )
                new_object_based_results = construct_object_based_dict(new_marked_results)
                new_object_based_results = remove_low_quality_objects(
                    new_object_based_results, min_frame_num=2
                )
                
                old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                def load_object_based_results(file_path):
                    """
                    Load the object based results from a json file.
                    :param file_path: The path to the json file
                    :return: The object based results
                    """
                    with open(file_path, "r") as f:
                        object_based_results = json.load(f)
                    # convert the string keys to integers
                    new_object_based_results = {}
                    for video_id, video_results in object_based_results.items():
                        new_video_results = {}
                        for object_id, object_results in video_results.items():
                            new_object_results = {}
                            for frame_id, frame_result in object_results.items():
                                new_object_results[int(frame_id)] = frame_result
                            new_video_results[int(object_id)] = new_object_results
                        new_object_based_results[int(video_id)] = new_video_results
                    return new_object_based_results
                old_object_based_results = load_object_based_results(
                    old_object_based_results_path
                )
                merged_object_based_results = merge_object_based_results(
                    old_object_based_results, new_object_based_results
                )
                merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                self.round_idx += 1
                with open(merged_objcet_based_results_path, "w") as f:
                    json.dump(merged_object_based_results, f, indent=4)
                merged_frame_based_results = construct_frame_based_dict(
                    merged_object_based_results
                )
                filtered_results = filter_marked_results(
                    merged_frame_based_results, min_frame_num=2
                )

                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))

                # self.resume_or_load(resume=False)
                # logger.info("Model reloaded for next round of training.")
                model = self._trainer.model
                new_optimizer = self.build_optimizer(cfg, model)
                self._trainer.optimizer = new_optimizer
                self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                logger.info("Optimizer and scheduler reloaded for next round of training.")
            
            elif self.test_mode == "ytvis_scoreiou_perc_adding_fra":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                if self.pred_iou_index >= len(self.pred_iou_percents):
                    self.pred_iou_index = len(self.pred_iou_percents) - 1
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                frame_scoreiou_percent = self.frame_score_iou_percents[self.frame_pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                if self.frame_pred_iou_index >= len(self.frame_score_iou_percents):
                    self.frame_pred_iou_index = len(self.frame_score_iou_percents) - 1
                if self.frame_pred_iou_index < len(self.frame_score_iou_percents) - 1:
                    self.frame_pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                
                metric_threshold = get_object_perc_metric_threshold(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                new_marked_results = mark_results(
                    results_dict, metric_threshold, min_frame_num=2
                )
                new_object_based_results = construct_object_based_dict(new_marked_results)
                new_object_based_results = remove_low_quality_objects(
                    new_object_based_results, min_frame_num=2
                )

                
                old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                def load_object_based_results(file_path):
                    """
                    Load the object based results from a json file.
                    :param file_path: The path to the json file
                    :return: The object based results
                    """
                    with open(file_path, "r") as f:
                        object_based_results = json.load(f)
                    # convert the string keys to integers
                    new_object_based_results = {}
                    for video_id, video_results in object_based_results.items():
                        new_video_results = {}
                        for object_id, object_results in video_results.items():
                            new_object_results = {}
                            for frame_id, frame_result in object_results.items():
                                new_object_results[int(frame_id)] = frame_result
                            new_video_results[int(object_id)] = new_object_results
                        new_object_based_results[int(video_id)] = new_video_results
                    return new_object_based_results
                old_object_based_results = load_object_based_results(
                    old_object_based_results_path
                )
                merged_object_based_results = merge_object_based_results(
                    old_object_based_results, new_object_based_results
                )
                merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                self.round_idx += 1
                with open(merged_objcet_based_results_path, "w") as f:
                    json.dump(merged_object_based_results, f, indent=4)
                merged_frame_based_results = construct_frame_based_dict(
                    merged_object_based_results
                )
                # filtered_results = filter_marked_results(
                #     merged_frame_based_results, min_frame_num=2
                # )
                new_metric_threshold = get_perc_metric_thresholdV2_min(
                    merged_frame_based_results, frame_scoreiou_percent
                )
                logger.info(f"frame metric threshold: {new_metric_threshold:.4f}")
                filtered_results = filter_metric_dictV2_min(
                    merged_frame_based_results, new_metric_threshold, min_frame_num=2
                )

                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))


            
            elif self.test_mode == "mose_scoreiou_perc_nms_minV2":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_thresholdV2_min(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dictV2_min(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/mose/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)
            
            elif self.test_mode == "mose_adding_refresh":
                if comm.is_main_process():
                    val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                    ap50 = val_results["segm"]["AP50"]
                    current_iou = ap50
                    logger.info("Current AP50: {:.4f}".format(ap50))
                    if ap50 > self.best_iou:
                        self.best_iou = ap50
                        # save the best model
                        self.checkpointer.save("model_best")
                        logger.info("Best model saved as model_best.pth")
                    inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                    inf_results = inf_results["total_results"]
                    if self.pred_iou_index >= len(self.pred_iou_percents):
                        self.pred_iou_index = len(self.pred_iou_percents) - 1
                    scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                    if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                        self.pred_iou_index += 1
                    results_dict = restruct_key_results(inf_results, "score_ious")
                    
                    metric_threshold = get_object_perc_metric_threshold(
                        results_dict, scoreiou_percent
                    )
                    logger.info(f"score_iou threshold: {metric_threshold:.4f}")

                    new_marked_results = mark_results(
                        results_dict, metric_threshold, min_frame_num=2
                    )
                    new_object_based_results = construct_object_based_dict(new_marked_results)
                    new_object_based_results = remove_low_quality_objects(
                        new_object_based_results, min_frame_num=2
                    )
                    
                    old_object_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx - 1}.json"
                    def load_object_based_results(file_path):
                        """
                        Load the object based results from a json file.
                        :param file_path: The path to the json file
                        :return: The object based results
                        """
                        with open(file_path, "r") as f:
                            object_based_results = json.load(f)
                        # convert the string keys to integers
                        new_object_based_results = {}
                        for video_id, video_results in object_based_results.items():
                            new_video_results = {}
                            for object_id, object_results in video_results.items():
                                new_object_results = {}
                                for frame_id, frame_result in object_results.items():
                                    new_object_results[int(frame_id)] = frame_result
                                new_video_results[int(object_id)] = new_object_results
                            new_object_based_results[int(video_id)] = new_video_results
                        return new_object_based_results
                    old_object_based_results = load_object_based_results(
                        old_object_based_results_path
                    )
                    merged_object_based_results = merge_object_based_results(
                        old_object_based_results, new_object_based_results
                    )
                    merged_objcet_based_results_path = f"OUTPUT-DIR/{self.model_name}/object_based_results_{self.round_idx}.json"
                    self.round_idx += 1
                    with open(merged_objcet_based_results_path, "w") as f:
                        json.dump(merged_object_based_results, f, indent=4)
                    merged_frame_based_results = construct_frame_based_dict(
                        merged_object_based_results
                    )
                    filtered_results = filter_marked_results(
                        merged_frame_based_results, min_frame_num=2
                    )

                    total_info = build_total_info(filtered_results)
                    frame_num = count_frame_num(total_info)
                    video_num = count_video_num(total_info)
                    object_num = len(total_info["annotations"])
                    logger.info("Frame number: {}".format(frame_num))
                    logger.info("Video number: {}".format(video_num))
                    logger.info("Object number: {}".format(object_num))
                    info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                    with open(info_path, "w") as f:
                        json.dump(total_info, f)
                comm.synchronize()

                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                self.resume_or_load(resume=False)
                logger.info("Model reloaded for next round of training.")
                model = self._trainer.model
                new_optimizer = self.build_optimizer(cfg, model)
                self._trainer.optimizer = new_optimizer
                self.scheduler = self.build_lr_scheduler(cfg, new_optimizer)
                logger.info("Optimizer and scheduler reloaded for next round of training.")
            
            elif self.test_mode == "ytvis_scoreiou_perc_nms_min_sw":
                val_results = self._last_eval_results[self.cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                
                inf_results = self._last_eval_results[self.cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info(f"Current Inference Dataset: {self.cfg.DATASETS.TEST[0]}")
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))
                inf_results = inf_results["total_results"]
                scoreiou_percent = self.pred_iou_percents[self.pred_iou_index]
                if self.pred_iou_index < len(self.pred_iou_percents) - 1:
                    self.pred_iou_index += 1
                results_dict = restruct_key_results(inf_results, "score_ious")
                metric_threshold = get_perc_metric_threshold_min(
                    results_dict, scoreiou_percent
                )
                logger.info(f"score_iou threshold: {metric_threshold:.4f}")
                filtered_results = filter_metric_dict_min(results_dict, metric_threshold)
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                object_num = len(total_info["annotations"])
                logger.info("Frame number: {}".format(frame_num))
                logger.info("Video number: {}".format(video_num))
                logger.info("Object number: {}".format(object_num))
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(self.cfg, self.model_name)
                if len(self.cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

                inf_ytvis = MyYTVIS(total_info, "datasets/ytvis_2019/train/JPEGImages")
                analysis_result = analysis_yvis(gt_yvis=self.gt_ytvis, inf_ytvis=inf_ytvis)
                gt_object_counts = analysis_result["gt_object_counts"]
                inf_object_counts = analysis_result["inf_object_counts"]
                intersection_counts = analysis_result["intersection_counts"]
                logger.info("GT object counts: {}".format(gt_object_counts))
                logger.info("INF object counts: {}".format(inf_object_counts))
                logger.info("Intersection counts: {}".format(intersection_counts))

                def switch_test_dataset_0(cfg, new_test_datasets):
                    cfg.defrost()
                    old_test_datasets = cfg.DATASETS.TEST
                    new_test_dataset_1 = old_test_datasets[1]
                    old_test_dataset_0 = old_test_datasets[0]
                    if old_test_dataset_0 == new_test_datasets[0]:
                        new_test_dataset_0 = new_test_datasets[1]
                    else:
                        assert old_test_dataset_0 == new_test_datasets[1]
                        new_test_dataset_0 = new_test_datasets[0]
                    cfg.DATASETS.TEST = (new_test_dataset_0, new_test_dataset_1)
                    cfg.freeze()
                    return cfg
                switch_test_dataset_0(self.cfg, self.test_datasets)

            elif self.test_mode == "ytvis_infer_score":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")

                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                results_dict = restruct_score_results(inf_results)
                logger.info(
                    "Saving inference results with score threshold: {:.2f}".format(
                        self.score_threshold
                    )
                )
                filtered_results = fileter_score_dict(
                    results_dict, self.score_threshold
                )
                total_info = build_total_info(filtered_results)
                frame_num = count_frame_num(total_info)
                video_num = count_video_num(total_info)
                logger.info(f"Frame number: {frame_num}")
                logger.info(f"Video number: {video_num}")
                info_path = f"datasets/ytvis_2019/{self.model_name}.json"
                with open(info_path, "w") as f:
                    json.dump(total_info, f)
                new_loader = self.get_train_loader(cfg, self.model_name)
                if len(cfg.DATASETS.TRAIN) > 1:
                    self._trainer.data_loader.loader1 = new_loader
                else:
                    self._trainer.data_loader = new_loader
                self._trainer._data_loader_iter_obj = iter(self._trainer.data_loader)

            elif self.test_mode == "frame_iou_pred_test":
                val_results = self._last_eval_results[cfg.DATASETS.TEST[1]]
                ap50 = val_results["segm"]["AP50"]
                current_iou = ap50
                logger.info("Current AP50: {:.4f}".format(ap50))
                if ap50 > self.best_iou:
                    self.best_iou = ap50
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                inf_results = self._last_eval_results[cfg.DATASETS.TEST[0]]["results"]
                pearson_score = inf_results["pearson"]
                spearman_score = inf_results["spearman"]
                logger.info("Current Pearson score: {:.4f}".format(pearson_score))
                logger.info("Current Spearman score: {:.4f}".format(spearman_score))

            else:
                raise NotImplementedError()

            if comm.is_main_process():
                current_iou = float(current_iou)
                logger.info("Current mean IoU: {:.4f}".format(current_iou))
                if current_iou > self.best_iou:
                    self.best_iou = current_iou
                    # save the best model
                    self.checkpointer.save("model_best")
                    logger.info("Best model saved as model_best.pth")
                    # print("Best model saved as model_best.pth")
                return {
                    "best_iou": self.best_iou,
                    "current_iou": current_iou,
                }
            else:
                return {}

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=10))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        if hasattr(cfg, "OUTPUT_RAW_MASKS") and cfg.OUTPUT_RAW_MASKS:
            logger.info("Output raw masks")
            model.output_raw_masks = True
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
If you want DefaultTrainer to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
"""
        )

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
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(
            round(cfg.SOLVER.WARMUP_ITERS / scale)
        )
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
