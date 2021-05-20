import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from backbone.resnets import resnet18_3d
from data.oai import MOAKSDataset
import os
import torch.nn.functional as F
from collections import defaultdict, deque
from argparse import ArgumentParser
from pathlib import Path
import json
from math import ceil


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_drop_step", type=int, default=50)
    parser.add_argument("--lr_drop_rate", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    return args


def _step(loader, epoch):
    total_steps = ceil(len(loader.dataset) / loader.batch_size)
    for step, batch in enumerate(loader):
        yield step, step + epoch * total_steps, batch


class MixCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def loss_labels(self, out, tgt):
        loss = F.binary_cross_entropy_with_logits(out, tgt.float())
        return loss

    def loss_boxes(self, out, tgt):
        loss = F.l1_loss(out, tgt.flatten(1))
        return loss

    def forward(self, out, tgt):
        losses = {}

        for key in ("boxes", "labels"):

            losses[key] = getattr(self, f"loss_{key}")(out[key], tgt[key])

        return losses


def main(args):

    root = "/scratch/visual/ashestak/oai/v00/data/"

    train_transforms = T.Compose(
        [T.ToTensor(), T.RandomResizedBBoxSafeCrop(), T.Normalize()]
    )

    val_transforms = T.Compose([T.ToTensor(), T.Normalize()])

    dataset_train = MOAKSDataset(
        os.path.join(root, "inputs"),
        os.path.join(root, "moaks", "train.json"),
        transforms=train_transforms,
    )
    # limit number of training images
    dataset_train.keys = dataset_train.keys[:4]
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset_val = MOAKSDataset(
        os.path.join(root, "inputs"),
        os.path.join(root, "moaks", "val.json"),
        transforms=val_transforms,
    )
    # limit number of val images
    dataset_val.keys = dataset_val.keys[:4]
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = resnet18_3d().to(device)
    criterion = MixCriterion().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )
    start = 0
    epochs = args.num_epochs
    window = args.window

    metrics = defaultdict(lambda: deque([], maxlen=window))

    if args.resume:

        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = checkpoint["epoch"] + 1

    output_dir = Path("/scratch/visual/ashestak/vit-pytorch/outputs/resnet18_3d_mri")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = SummaryWriter()

    best_val_loss = np.inf

    weight_dict = {"boxes": 5, "labels": 1}

    train_steps = ceil(len(dataset_train) / dataloader_train.batch_size)
    val_steps = ceil(len(dataset_val) / dataloader_val.batch_size)

    for epoch in range(start, epochs):

        total_loss = 0
        total_steps = 0

        for step, (img, tgt) in enumerate(dataloader_train):

            global_step = step + epoch * train_steps

            out = model(img)

            loss_dict = criterion(out, tgt)
            loss = sum(weight_dict[k] * loss_dict[k] for k in weight_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().cpu().item()

            metrics["loss"].append(loss_value)
            for key, val in loss_dict.items():
                metrics[key] = val.detach().item()

            print(step)
            print(step % window)

            if step and (step % window == 0):
                print(metrics)
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

            for step, (img, tgt) in enumerate(dataloader_val):

                global_step = step + epoch * val_steps

                out = model(img)

                loss_dict = criterion(out, tgt)
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
