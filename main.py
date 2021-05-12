from argparse import ArgumentParser
from vit_pytorch.vit import ViT
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.mri import build_mri, build_dicomu


def parse_args(parser: ArgumentParser):

    # * Dataset

    parser.add_argument(
        "--root", type=str, default="/scratch/htc/ashestak/oai/v00/data"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)

    # * Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # * Outputs
    parser.add_argument(
        "--output_dir", type=str, default="/scratch/visual/ashestak/outputs/vit"
    )

    # * Logging
    parser.add_argument(
        "--log_dir", type=str, default="/scratch/visual/ashestak/logging/vit"
    )

    # * VIT
    parser.add_argument("--image_size", nargs="+", type=int, default=[160, 384, 384])
    parser.add_argument("--patch_size", nargs="+", type=int, default=[16, 32, 32])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256, help="Dimension of the encoder")
    parser.add_argument(
        "--depth", type=int, default=6, help="Depth of the encoder stack"
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--pool", type=str, default="cls")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--emb_dropout", type=float, default=0.0)

    # * DINO

    return parser.parse_args()


def to_device(img, tgt=None, device="cpu"):

    img = img.to(device)

    if tgt is not None:
        tgt = [{k: v.to(device) for k, v in t.items()} for t in tgt]

    return img, tgt


def main(args):
    vit = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        pool=args.pool,
        channels=args.channels,
        dim_head=args.dim_head,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    )
    vit.to(args.device)

    optimizer = torch.optim.Adam(vit.parameters(), args.lr)
    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_worker=args.num_workers,
    )
    logger = SummaryWriter(log_dir=args.log_dir)

    print("Starting training")
    for epoch in range(args.start_epoch, args.epochs):
        for img, tgt in tqdm(
            dataloader_train, desc=f"Train (Ep. {epoch:03d}/{args.epochs:03d})"
        ):

            img, tgt = to_device(img, tgt, args.device)

            print(img.shape)
            print(tgt)
            break

            loss = None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

        with torch.no_grad():

            predictions = []
            targets = []
            for img, tgt in tqdm(dataloader_val, desc="Validation"):

                preds = vit(img)
                predictions.append(preds)
                targets.append(tgt)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parse_args(parser)
    main(args)
