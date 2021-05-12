import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from backbone.resnets import resnet18_3d
from medical.data.oai import DICOMDatasetMasks
import os
import torch.nn.functional as F
from collections import deque
from argparse import ArgumentParser
from pathlib import Path
import json
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_drop_step", type=int, default=50)
    parser.add_argument("--lr_drop_rate", type=float, default=0.5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    return args


def main(args):
    root_data = "/scratch/htc/bzftacka/"
    root_anns = "/scratch/htc/ashestak/oai/v00/moaks/"

    dataset_train = DICOMDatasetMasks(root_data, os.path.join(root_anns, "train.json"))
    print("Training data: ", len(dataset_train), flush=True)

    dataset_val = DICOMDatasetMasks(root_data, os.path.join(root_anns, "val.json"))
    print("Validation data: ", len(dataset_val), flush=True)
    model = resnet18_3d()
    
    print(model, flush=True)

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop_step, args.lr_drop_rate
    )

    start = 0

    if args.resume:
        ckpt = torch.load(args.resume)
        print("resuming from ", args.resume)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1

    epochs = args.num_epochs
    window = args.window
    windowed_loss = deque(maxlen=window)

    output_dir = Path("/scratch/htc/ashestak/vit-pytorch/outputs/resnet18_3d")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = SummaryWriter()

    best_val_loss = np.inf

    global_step = 0

    for epoch in range(start, epochs):
        
        total_loss = 0
        num_steps = 0

        print(f"Epoch {epoch:03d}/{epochs:03d}", flush=True)
        t0 = time.time()
        for step, (img, tgt) in enumerate(dataloader_train):

            img = img.float().to(device)
            tgt = tgt.to(device)

            out = model(img)

            loss = F.l1_loss(out.sigmoid(), tgt.flatten(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().item()
            windowed_loss.append(loss_value)

            if global_step and (global_step % window == 0):

                logger.add_scalar("loss", np.mean(windowed_loss), global_step=global_step)
                # logger.add_scalar("lr", optimizer.lr, global_step=global_step)

            total_loss += loss_value
            num_steps += 1
            global_step += 1
        
        total_loss = total_loss / num_steps
        logger.add_scalar("train_loss_epoch", total_loss, global_step=epoch)

        print(f"Time {(time.time() - t0):.4f} || Loss: {total_loss:.4f}")

        with torch.no_grad():

            model.eval()

            total_loss = 0
            num_steps = 0

            for step, (img, tgt) in enumerate(dataloader_val):

                img = img.float().to(device)
                tgt = tgt.to(device)
                out = model(img)

                loss = F.l1_loss(out.sigmoid(), tgt.flatten(1))
                total_loss += loss.item()
                num_steps += 1

            logger.add_scalar(
                "val_loss_epoch", total_loss / num_steps, global_step=epoch
            )

            if (best_loss := total_loss / num_steps) < best_val_loss:
                best_val_loss = best_loss

                torch.save(model.state_dict(), output_dir / "best_model.pt")

                with open(output_dir / "best_model.json", "w") as fh:
                    json.dump({"epoch": epoch, "val_loss": best_loss}, fh)

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
