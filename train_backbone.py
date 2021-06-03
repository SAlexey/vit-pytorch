# %%

import json
import os
from argparse import ArgumentParser
from collections import defaultdict, deque
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backbone.resnets import Net1, Net2, resnet18_3d, resnet50_3d
from data import transforms as T
from data.oai import MOAKSDataset, MOAKSDatasetBinaryMultilabel


# %%


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_drop_step", type=int, default=50)
    parser.add_argument("--lr_drop_rate", type=float, default=0.5)
    parser.add_argument(
        "--resume", type=str, default="", help="checkpoinnt path to resume from"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=("resnet18_3d", "resnet50_3d"),
        default="resnet18_3d",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="path to model weights (ignores 'model' key in checkpoint)",
    )
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument(
        "--data_dir", type=str, default="/scratch/htc/ashestak/oai/v00/data/inputs"
    )
    parser.add_argument(
        "--anns_dir", type=str, default="/scratch/htc/ashestak/oai/v00/data/moaks"
    )

    parser.add_argument("--output_dir", type=str, default="outputs/backbone")
    args = parser.parse_args()
    return args


class MixCriterion(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def loss_labels(self, out, tgt, label_pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(out, tgt, pos_weight=label_pos_weight)
        return loss

    def loss_boxes(self, out, tgt):
        loss = F.l1_loss(out, tgt)
        return loss

    def forward(self, out, tgt, label_pos_weight=None):
        losses = {}
        kwargs = {"labels": {"label_pos_weight": label_pos_weight}}

        for key in ("labels", "boxes"):
            loss_func = getattr(self, f"loss_{key}")
            loss_kwargs = kwargs.get(key, dict())
            losses[key] = loss_func(out[key], tgt[key], **loss_kwargs)

        return losses


def main(args):

    root = Path(args.data_dir)
    anns = Path(args.anns_dir)

    assert root.exists(), "Provided data directory doesn't exist!"
    assert anns.exists(), "Provided annotations directory doesn't exist!"

    train_transforms = T.Compose(
        [T.ToTensor(), T.RandomResizedBBoxSafeCrop(), T.Normalize()]
    )

    val_transforms = T.Compose([T.ToTensor(), T.Normalize()])

    dataset_train = MOAKSDatasetBinaryMultilabel(
        root,
        anns / "train.json",
        transforms=train_transforms,
    )
    # limit number of training images
    # dataset_train.keys = dataset_train.keys[:4]
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset_val = MOAKSDatasetBinaryMultilabel(
        root,
        anns / "val.json",
        transforms=val_transforms,
    )
    # limit number of val images
    # dataset_val.keys = dataset_val.keys[:4]
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    # TODO:  better network initializationwith args
    model = Net2(args.backbone, dropout=args.dropout).to(device)
    criterion = MixCriterion().to(device)

    # criterion = F.binary_cross_entropy_with_logits

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    start = 0
    epochs = args.num_epochs
    window = args.window

    metrics = defaultdict(lambda: deque([], maxlen=window))

    if args.weights:
        state_dict = torch.load(args.weights)
        model.load_state_dict(state_dict)

    if args.resume:

        checkpoint = torch.load(args.resume)
        if not args.weights:
            model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = checkpoint["epoch"] + 1

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = np.inf

    weight_dict = {"labels": 1, "boxes": 5}

    train_steps = ceil(len(dataset_train) / dataloader_train.batch_size)
    val_steps = ceil(len(dataset_val) / dataloader_val.batch_size)

    logger = SummaryWriter()
    for epoch in range(start, epochs):
        print(f"Epoch {epoch:03d}/{epochs:03d}", flush=True)

        total_loss = 0
        total_steps = 0

        pos_weight = dataloader_train.dataset.pos_weight
        if isinstance(pos_weight, torch.Tensor):
            pos_weight = pos_weight.to(device)
        model.train()

        for step, (img, tgt) in enumerate(dataloader_train):

            img = img.to(device)
            tgt = {k: v.to(device).float() for k, v in tgt.items()}

            global_step = step + epoch * train_steps

            out = model(img)

            loss_dict = criterion(
                out,
                tgt,
                label_pos_weight=pos_weight,
            )
            loss = sum(weight_dict[k] * loss_dict[k] for k in weight_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().cpu().item()

            metrics["loss"].append(loss_value)
            for key, val in loss_dict.items():
                metrics[key] = val.detach().cpu().item()

            if step and (step % window == 0):
                for key, val in metrics.items():
                    logger.add_scalar(key, np.mean(val), global_step=global_step)

            total_loss += loss_value
            total_steps += 1

        logger.add_scalar(
            "train_loss_epoch", total_loss / total_steps, global_step=epoch
        )

        with torch.no_grad():

            model.eval()

            total_loss = 0
            total_steps = 0

            pos_weight = dataloader_val.dataset.pos_weight
            if isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.to(device)

            for step, (img, tgt) in enumerate(dataloader_val):

                img = img.to(device)
                tgt = {k: v.to(device).float() for k, v in tgt.items()}

                global_step = step + epoch * val_steps

                out = model(img)

                loss_dict = criterion(
                    out,
                    tgt,
                    label_pos_weight=pos_weight,
                )
                loss_value = sum(weight_dict[k] * loss_dict[k] for k in weight_dict)

                total_loss += loss_value
                total_steps += 1

            epoch_loss = total_loss / total_steps

            logger.add_scalar("val_loss_epoch", epoch_loss, global_step=epoch)

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss

                torch.save(model.state_dict(), output_dir / "best_model.pt")

                with open(output_dir / "best_model.json", "w") as fh:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss.item()}, fh)

        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            output_dir / "checkpoint.ckpt",
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)

# %%
