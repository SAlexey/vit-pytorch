import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from backbone.resnets import resnet18_3d
from data.mri import DICOMDatasetMasks
import os
import torch.nn.functional as F
from collections import deque


def main():

    root = "/scratch/visual/ashestak/oai/v00/data/"

    dataset_train = DICOMDatasetMasks(
        root, os.path.join(root, "annotations", "train.json")
    )
    dataset_val = DICOMDatasetMasks(root, os.path.join(root, "annotations", "val.json"))

    model = resnet18_3d()

    dataloader_train = DataLoader(
        dataset_train, shuffle=True, batch_size=1, num_workers=3
    )
    dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=1, num_workers=3)

    logger = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

    epochs = 100
    window = 20
    windowed_loss = deque(maxlen=window)

    for epoch in range(epochs):

        loader = tqdm(dataloader_train)
        loader.set_description(f"Epoch {epoch:03d}/{epochs:03d}")

        for step, (img, tgt) in enumerate(loader):

            img = img.float()
            out = model(img)

            loss = F.l1_loss(out.sigmoid(), tgt.flatten(1))

            windowed_loss.append(loss.detach().item())

            if step % 20 == 0:
                loader.set_postfix({"loss": np.mean(windowed_loss)})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        break


if __name__ == "__main__":
    main()
