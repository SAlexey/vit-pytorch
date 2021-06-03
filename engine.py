import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm


def training_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch,
    progress=False,
    logger=None,
    window=20,
    postprocess=None,
):

    global_step = 0
    global_loss = 0
    total_steps = 0
    running_loss = deque(maxlen=window)

    model.train()

    if progress:
        loader = tqdm(loader)

    for ins, tgt in loader:
        out = model(ins)

        if postprocess is not None:
            ins, out, tgt = postprocess(ins, out, tgt=tgt)

        loss = criterion(out, tgt)
        running_loss.append(loss.item())
