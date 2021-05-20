import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from backbone.resnets import resnet18_3d
from data.mri import DICOMDatasetMasks
import os
import torch.nn.functional as F
from collections import deque
from argparse import ArgumentParser
from pathlib import Path
import json

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/htc/ashestak/oai/v00/data")


def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
