#!/usr/bin/env python
"""Phase 1: Encoder distillation pretraining on DataComp-12M."""

import sys
from pathlib import Path

import torch

from callbacks import beta2_override, grad_clip, wandb_config
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train_image_encoder import ImageEncoderTrainer

RECIPES = {
    "default": dict(lr0=3e-4, weight_decay=0.05, warmup_epochs=1, epochs=10, momentum=0.9, grad_clip=3.0, beta2=None),
    "unic": dict(lr0=6e-4, weight_decay=0.03, warmup_epochs=2, epochs=30, momentum=0.9, grad_clip=None, beta2=0.99),
    "radio": dict(lr0=1e-3, weight_decay=0.02, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=1.0, beta2=0.95),
}


def _pop_resume(argv: list[str]) -> tuple[list[str], str]:
    """Return argv without '--resume <path>' and the resume path."""
    if "--resume" not in argv:
        return argv, ""
    index = argv.index("--resume")
    return argv[:index] + argv[index + 2 :], argv[index + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


def main(argv: list[str]) -> None:
    """Launch a fresh phase 1 run or resume from a checkpoint.

    Args:
        argv: [gpu, teachers, name, recipe, model_yaml]
        recipe: "default", "unic", or "radio"
        model_yaml: e.g. "yolo26s-cls.yaml" or "yolo26l-cls.yaml"
    """
    argv, resume = _pop_resume(argv[1:])
    resume_args = _load_train_args(resume) if resume else {}
    gpu = argv[0] if argv else "0"
    teachers = argv[1] if len(argv) > 1 else resume_args.get("teachers", "eupe:vitb16")
    name = (
        argv[2] if len(argv) > 2 else resume_args.get("name", f"phase1-{teachers.replace(':', '-').replace('+', '_')}")
    )
    recipe = argv[3] if len(argv) > 3 else "default"
    model_yaml = argv[4] if len(argv) > 4 else "yolo26s-cls.yaml"
    r = RECIPES[recipe]

    model = YOLO(model_yaml)
    if r["grad_clip"]:
        model.add_callback("on_train_start", grad_clip.override(r["grad_clip"]))
    if r["beta2"]:
        model.add_callback("on_train_start", beta2_override.override(r["beta2"]))
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            teachers=teachers,
            recipe=recipe,
            cos_weight=0.9,
            l1_weight=0.1,
            grad_clip=r["grad_clip"],
            beta2=r["beta2"],
            wandb_group="distill",
        ),
    )
    train_args = dict(
        trainer=ImageEncoderTrainer,
        teachers=teachers,
        data="/data/shared-datasets/datacomp-12m",
        device=gpu,
        project=resume_args.get("project", "yolo-next-encoder"),
        name=name,
        epochs=r["epochs"],
        batch=128,
        imgsz=224,
        patience=5,
        nbs=512,
        cos_lr=True,
        lr0=r["lr0"],
        lrf=0.01,
        momentum=r["momentum"],
        weight_decay=r["weight_decay"],
        warmup_epochs=r["warmup_epochs"],
        warmup_bias_lr=0,
        dropout=0,
        optimizer="AdamW",
        pretrained=False,
        amp=True,
        seed=0,
        deterministic=True,
        fliplr=0.5,
        workers=8,
    )
    if resume:
        train_args["resume"] = resume
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
