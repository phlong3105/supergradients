#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket

import click

from mon import core
from my_utils import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val, parse_dataset_args,
    parse_detection_yolo_training_params,
)
from super_gradients import Trainer
from super_gradients.training import models
from super_gradients.training.utils.checkpoint_utils import load_pretrained_weights_local

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(opt: argparse.Namespace):
    root       = opt.root
    task       = opt.task
    weights    = opt.weights
    weights    = weights[0] if isinstance(weights, list | tuple) else weights
    weights    = core.Path(weights)
    model      = opt.model
    data       = opt.data
    save_dir   = opt.save_dir
    batch_size = opt.batch_size
    workers    = opt.workers
    imgsz      = opt.imgsz
    imgsz      = [imgsz, imgsz] if isinstance(imgsz, int) else imgsz
    
    # Define dataset
    dataset_params = parse_dataset_args(data)
    train_data     = coco_detection_yolo_format_train(
        dataset_params    = {
            "data_dir"  : dataset_params["data_dir"],
            "images_dir": dataset_params["train_images_dir"],
            "labels_dir": dataset_params["train_labels_dir"],
            "classes"   : dataset_params["classes"],
            "input_dim" : imgsz,
        },
        dataloader_params = {
            "batch_size" : batch_size,
            "num_workers": workers,
        }
    )
    val_data       = coco_detection_yolo_format_val(
        dataset_params    = {
            "data_dir"  : dataset_params["data_dir"],
            "images_dir": dataset_params["val_images_dir"],
            "labels_dir": dataset_params["val_labels_dir"],
            "classes"   : dataset_params["classes"],
            "input_dim" : imgsz,
        },
        dataloader_params = {
            "batch_size" : batch_size,
            "num_workers": workers,
        }
    )
    test_data      = coco_detection_yolo_format_val(
        dataset_params    = {
            "data_dir"  : dataset_params["data_dir"],
            "images_dir": dataset_params["test_images_dir"],
            "labels_dir": dataset_params["test_labels_dir"],
            "classes"   : dataset_params["classes"],
            "input_dim" : imgsz,
        },
        dataloader_params = {
            "batch_size" : batch_size,
            "num_workers": workers,
        }
    )
    
    # Define model
    if weights.is_ckpt_file():
        net = models.get(model, num_classes=len(dataset_params["classes"]), checkpoint_path=weights)
    elif weights.is_weights_file():
        net = models.get(model, num_classes=len(dataset_params["classes"]))
        load_pretrained_weights_local(net, model, weights)
    else:
        net = models.get(model, num_classes=len(dataset_params["classes"]))
    
    # Define trainer
    if task in ["detect"]:
        training_params = parse_detection_yolo_training_params(vars(opt), dataset_params)
    else:
        raise ValueError(f"Invalid task: {task}")
    
    trainer = Trainer(experiment_name=str(save_dir.name), ckpt_root_dir=str(save_dir.parent))
    trainer.train(model=net, training_params=training_params, train_loader=train_data, valid_loader=val_data)

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root      or args.get("root")
    weights  = weights   or args.get("weights")
    model    = model     or args.get("model")
    data     = args.get("data")
    project  = args.get("project")
    fullname = fullname  or args.get("name")
    device   = device    or args.get("device")
    epochs   = epochs    or args.get("epochs")
    exist_ok = exist_ok  or args.get("exist_ok")
    verbose  = verbose   or args.get("verbose")
    
    # Parse arguments
    root     = core.Path(root)
    weights  = core.to_list(weights)
    model    = str(model)
    data     = core.Path(data)
    data     = data if data.exists() else _current_dir / "data" / data.name
    data     = str(data.config_file())
    project  = root.name or project
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = core.Path(save_dir)
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["root"]       = root
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
    # args["epochs"]     = epochs
    args["max_epochs"] = epochs
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    
    opt = argparse.Namespace(**args)
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(opt.save_dir))
    core.Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Call train()
    train(opt=opt)
    
    return str(opt.save_dir)
        

if __name__ == "__main__":
    main()

# endregion
