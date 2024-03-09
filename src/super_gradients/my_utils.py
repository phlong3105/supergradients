#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "MyYoloDarknetFormatDetectionDataset",
    "coco_detection_yolo_format_train",
    "coco_detection_yolo_format_val",
    "parse_dataset_args",
    "parse_detection_yolo_training_params",
]

import os

import yaml
from torch.utils.data import DataLoader

from mon import core, DATA_DIR
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Dataloaders
from super_gradients.common.registry import register_dataloader
from super_gradients.training.dataloaders import get_data_loader
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import \
    YoloDarknetFormatDetectionDataset
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.params import TrainingParams
from supergradients.src.super_gradients.training.utils.media.image import is_image

logger        = get_logger(__name__)
console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


class MyYoloDarknetFormatDetectionDataset(YoloDarknetFormatDetectionDataset):
    
    def _setup_data_source(self) -> int:
        if isinstance(self.images_dir, str | core.Path):
            self.images_dir = [self.images_dir]
        if isinstance(self.labels_dir, str | core.Path):
            self.labels_dir = [self.labels_dir]
        
        self.data_dir      = core.Path(self.data_dir)
        self.images_folder = [self.data_dir / x for x in self.images_dir]
        self.labels_folder = [self.data_dir / x for x in self.labels_folder]
        
        all_images_file_names = []
        all_labels_file_names = []
        for images_f, labels_f in zip(self.images_folder, self.labels_folder):
            all_images_file_names += list(image_name for image_name in os.listdir(images_f) if is_image(image_name))
            all_labels_file_names += list(label_name for label_name in os.listdir(labels_f) if label_name.endswith(".txt"))
            
        remove_file_extension = lambda file_name: os.path.splitext(os.path.basename(file_name))[0]
        unique_image_file_base_names = set(remove_file_extension(image_file_name) for image_file_name in all_images_file_names)
        unique_label_file_base_names = set(remove_file_extension(label_file_name) for label_file_name in all_labels_file_names)
       
        images_not_in_labels = unique_image_file_base_names - unique_label_file_base_names
        if images_not_in_labels:
            logger.warning(f"{len(images_not_in_labels)} images are note associated to any label file")

        labels_not_in_images = unique_label_file_base_names - unique_image_file_base_names
        if labels_not_in_images:
            logger.warning(f"{len(labels_not_in_images)} label files are not associated to any image.")

        # Only keep names that are in both the images and the labels
        valid_base_names = unique_image_file_base_names & unique_label_file_base_names
        if len(valid_base_names) != len(all_images_file_names):
            logger.warning(
                f"As a consequence, "
                f"{len(valid_base_names)}/{len(all_images_file_names)} images and "
                f"{len(valid_base_names)}/{len(all_labels_file_names)} label files will be used."
            )
        
        self.images_file_names = []
        self.labels_file_names = []
        for image_full_name in all_images_file_names:
            base_name = remove_file_extension(image_full_name)
            if base_name in valid_base_names:
                self.images_file_names.append(image_full_name)
                self.labels_file_names.append(base_name + ".txt")
        return len(self.images_file_names)


@register_dataloader(Dataloaders.COCO_DETECTION_YOLO_FORMAT_TRAIN)
def coco_detection_yolo_format_train(dataset_params: dict = None, dataloader_params: dict = None) -> DataLoader:
    return get_data_loader(
        config_name       = "coco_detection_yolo_format_base_dataset_params",
        dataset_cls       = MyYoloDarknetFormatDetectionDataset,
        train             = True,
        dataset_params    = dataset_params,
        dataloader_params = dataloader_params,
    )


@register_dataloader(Dataloaders.COCO_DETECTION_YOLO_FORMAT_VAL)
def coco_detection_yolo_format_val(dataset_params: dict = None, dataloader_params: dict = None) -> DataLoader:
    return get_data_loader(
        config_name       = "coco_detection_yolo_format_base_dataset_params",
        dataset_cls       = MyYoloDarknetFormatDetectionDataset,
        train             = False,
        dataset_params    = dataset_params,
        dataloader_params = dataloader_params,
    )


def parse_dataset_args(data: str | core.Path) -> dict:
    assert isinstance(data, dict | str | core.Path), f"Invalid data file: {data}"
    
    if isinstance(data, str | core.Path):
        assert core.Path(data).is_yaml_file(), f"Invalid data file: {data}"
        with open(data, errors="ignore") as f:
            data = yaml.safe_load(f)
    
    for k in ["train", "val", "names"]:
        assert k in data, f"data.yaml '{k}:' field missing ❌"
    
    if isinstance(data["names"], dict):
        data["names"] = list(data["names"].values())
    
    path = core.Path(data.get("path", ""))
    path = DATA_DIR / path
    assert path.exists(), f"Invalid data path: {path}"
    for k in ["train", "val", "names"]:
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                data[f"{k}_images_dir"] = str(data[k])
                data[f"{k}_labels_dir"] = str(data[k]).replace("/images/", "/labels/")
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
                data[f"{k}_images_dir"] = [str(x) for x in data[k]]
                data[f"{k}_labels_dir"] = [str(x).replace("/images/", "/labels/") for x in data[k]]
                
    dataset_params = {
        "data_dir"        : path,
        "train_images_dir": data["train_labels_dir"],
        "train_labels_dir": data["train_labels_dir"],
        "val_images_dir"  : data["val_images_dir"],
        "val_labels_dir"  : data["val_labels_dir"],
        "test_images_dir" : data["test_images_dir"],
        "test_labels_dir" : data["test_labels_dir"],
        "classes"         : data["names"],
    }
    return dataset_params


def parse_detection_yolo_training_params(
    train_params  : dict,
    dataset_params: dict,
) -> dict:
    training_params  = TrainingParams()
    training_params.override(**train_params)
    training_params |= {
        "loss": PPYoloELoss(
            use_static_assigner = False,
            num_classes         = len(dataset_params["classes"]),
            reg_max             = 16,
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres       = 0.1,
                top_k_predictions = 300,
                # NOTE: num_classes needs to be defined here
                num_cls           = len(dataset_params["classes"]),
                normalize_targets = True,
                post_prediction_callback = PPYoloEPostPredictionCallback(
                    score_threshold = 0.01,
                    nms_top_k       = 1000,
                    max_predictions = 300,
                    nms_threshold   = 0.7
                )
            ),
            DetectionMetrics_050_095(
                score_thres       = 0.1,
                top_k_predictions = 300,
                # NOTE: num_classes needs to be defined here
                num_cls           = len(dataset_params["classes"]),
                normalize_targets = True,
                post_prediction_callback = PPYoloEPostPredictionCallback(
                    score_threshold = 0.01,
                    nms_top_k       = 1000,
                    max_predictions = 300,
                    nms_threshold   = 0.7
                )
            )
        ],
        "metric_to_watch": "mAP@0.50"
    }
    
    return training_params
