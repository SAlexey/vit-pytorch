#%%
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
from einops import rearrange, parse_shape, repeat
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import matplotlib.pyplot as plt
import nibabel as nib
import sys

#%%
def main():

    model = resnet18_3d()
    state_dict = torch.load(
        "/scratch/htc/ashestak/vit-pytorch/outputs/resnet18_3d/best_model.pt"
    )
    model.load_state_dict(state_dict)
    model.eval()

    anns = "/scratch/htc/ashestak/oai/v00/data/moaks/test.json"
    root = "/scratch/htc/bzftacka"
    data = DICOMDatasetMasks(root, anns)

    loader = DataLoader(data, batch_size=2, num_workers=2)

    output_dir = "/scratch/htc/ashestak/vit-pytorch/outputs/resnet18_3d"

    assert os.path.exists(output_dir)

    boxes_tgt = []
    boxes_out = []
    boxes_iou = []
    image_ids = []

    i = 0
    with torch.no_grad():
        for img, tgt in loader:

            if i == 10:
                break

            out_box = model(img.float()).sigmoid()
            tgt_box = tgt["boxes"]
            tgt_ids = tgt["image_id"]

            shape = parse_shape(tgt_box, "batch boxes coord")

            out_box = rearrange(
                out_box, "batch (boxes coord) -> (batch boxes) coord", **shape
            )
            tgt_box = rearrange(
                tgt_box, "batch boxes coord -> (batch boxes) coord", **shape
            )
            shape.pop("coord")
            tgt_ids = repeat(tgt_ids, "(batch ids) -> (batch boxes ids)", **shape)

            size = tuple(img.shape[-3:])
            size = torch.as_tensor(size + size)

            tgt_xyxy = box_cxcywh_to_xyxy(tgt_box)
            out_xyxy = box_cxcywh_to_xyxy(out_box)

            ious = box_iou(tgt_xyxy, out_xyxy).diag()

            tgt_box = (tgt_xyxy * size).round().int()
            out_box = (out_xyxy * size).round().int()

            boxes_tgt.append(tgt_box)
            boxes_out.append(out_box)
            boxes_iou.append(ious)
            image_ids.append(tgt_ids)

            i += 1

    boxes_out = rearrange(boxes_out, "list boxes coord -> (list boxes) coord")
    boxes_tgt = rearrange(boxes_tgt, "list boxes coord -> (list boxes) coord")
    boxes_iou = rearrange(boxes_iou, "list ious -> (list ious)")
    image_ids = rearrange(image_ids, "list ids -> (list ids)")

    np.savez(
        "test_results.npz",
        {
            "out_boxes": boxes_out.cpu().numpy(),
            "tgt_boxes": boxes_tgt.cpu().numpy(),
            "ious": boxes_iou.cpu().numpy(),
            "image_ids": image_ids.cpu().numpy(),
        },
    )

    sys.exit(1)

    iou_ind_asc = torch.argsort(boxes_iou)

    num_w = 5  # number of worst examples
    num_b = 5  # number of best examples

    worst = iou_ind_asc[:num_w]
    best = iou_ind_asc[-num_b:]

    gs_kw = {
        "width_ratios": [2, 2, 2, 1, 1, 1, 1, 1, 1],
        "height_ratios": [
            1,
        ]
        * num_w,
    }
    _, axes = plt.subplots(
        num_w,
        len(gs_kw["width_ratios"]),
        figsize=(20, 20),
        constrained_layout=True,
        gridspec_kw=gs_kw,
    )

    for idx, ax_row in zip(worst, axes):

        tgt = boxes_tgt[idx]
        out = boxes_out[idx]
        iou = boxes_iou[idx]
        key = image_ids[idx]

        ind = loader.dataset.keys.index(str(key.item()))
        img, _ = loader.dataset[ind]
        img = img.squeeze()

        img_slices = []
        box_slices = []

        min = torch.min(tgt, out)[0]
        max = torch.max(tgt, out)[-3]
        slices = np.linspace(min, max, 3).astype(int)

        for slice in slices:
            box_slices.append((out[[2, 1, 5, 4]], tgt[[2, 1, 5, 4]]))
            img_slices.append(img[slice])

        min = torch.min(tgt, out)[1]
        max = torch.max(tgt, out)[-2]
        slices = np.linspace(min, max, 3).astype(int)
        for slice in slices:
            box_slices.append((out[[2, 0, 5, 3]], tgt[[2, 0, 5, 3]]))
            img_slices.append(img[:, slice])

        min = torch.min(tgt, out)[2]
        max = torch.max(tgt, out)[-1]
        slices = np.linspace(min, max, 3).astype(int)
        for slice in slices:
            box_slices.append((out[[1, 0, 3, 2]], tgt[[1, 0, 3, 2]]))
            img_slices.append(img[:, :, slice])

        for (
            ax,
            img_slice,
            (
                out_box_slice,
                tgt_box_slice,
            ),
        ) in zip(ax_row, img_slices, box_slices):

            ax.imshow(img_slice, "gray")
            ax.add_patch(
                plt.Rectangle(
                    tgt_box_slice[:2],
                    *(tgt_box_slice[2:] - tgt_box_slice[:2]),
                    fill=False,
                    ec="blue",
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    out_box_slice[:2],
                    *(out_box_slice[2:] - out_box_slice[:2]),
                    fill=False,
                    ec="red",
                )
            )
            ax.axis("off")
    fname = os.path.join(output_dir, f"Worst_{num_w}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    gs_kw = {
        "width_ratios": [2, 2, 2, 1, 1, 1, 1, 1, 1],
        "height_ratios": [
            1,
        ]
        * num_b,
    }
    _, axes = plt.subplots(
        num_b,
        len(gs_kw["width_ratios"]),
        figsize=(20, 20),
        constrained_layout=True,
        gridspec_kw=gs_kw,
    )

    for idx, ax_row in zip(best, axes):

        tgt = boxes_tgt[idx]
        out = boxes_out[idx]
        iou = boxes_iou[idx]
        key = image_ids[idx]

        ind = loader.dataset.keys.index(str(key.item()))
        img, _ = loader.dataset[ind]
        img = img.squeeze()

        img_slices = []
        box_slices = []

        min = torch.min(tgt, out)[0]
        max = torch.max(tgt, out)[-3]
        slices = np.linspace(min, max, 3).astype(int)

        for slice in slices:
            box_slices.append((out[[2, 1, 5, 4]], tgt[[2, 1, 5, 4]]))
            img_slices.append(img[slice])

        min = torch.min(tgt, out)[1]
        max = torch.max(tgt, out)[-2]
        slices = np.linspace(min, max, 3).astype(int)
        for slice in slices:
            box_slices.append((out[[2, 0, 5, 3]], tgt[[2, 0, 5, 3]]))
            img_slices.append(img[:, slice])

        min = torch.min(tgt, out)[2]
        max = torch.max(tgt, out)[-1]
        slices = np.linspace(min, max, 3).astype(int)
        for slice in slices:
            box_slices.append((out[[1, 0, 3, 2]], tgt[[1, 0, 3, 2]]))
            img_slices.append(img[:, :, slice])

        for (
            ax,
            img_slice,
            (
                out_box_slice,
                tgt_box_slice,
            ),
        ) in zip(ax_row, img_slices, box_slices):

            ax.imshow(img_slice, "gray")
            ax.add_patch(
                plt.Rectangle(
                    tgt_box_slice[:2],
                    *(tgt_box_slice[2:] - tgt_box_slice[:2]),
                    fill=False,
                    ec="blue",
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    out_box_slice[:2],
                    *(out_box_slice[2:] - out_box_slice[:2]),
                    fill=False,
                    ec="red",
                )
            )
            ax.axis("off")
    fname = os.path.join(output_dir, f"Best_{num_b}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
