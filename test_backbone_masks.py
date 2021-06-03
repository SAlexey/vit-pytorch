#%%
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from backbone.resnets import resnet18_3d
from data.oai import DICOMDatasetMasks
import os
import torch.nn.functional as F
from einops import rearrange, parse_shape, repeat
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import matplotlib.pyplot as plt
import sys

#%%
def main():

    model = resnet18_3d()
    state_dict = torch.load(
        "/scratch/visual/ashestak/vit-pytorch/outputs/cluster/resnet18_3d/best_model.pt"
    )
    model.load_state_dict(state_dict)
    model.eval()

    anns = "/scratch/visual/ashestak/oai/v00/data/moaks/test.json"
    root = "/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/"
    data = DICOMDatasetMasks(root, anns)

    loader = DataLoader(data, batch_size=1, num_workers=2)

    output_dir = "/scratch/visual/ashestak/vit-pytorch/outputs/resnet18_3d"

    assert os.path.exists(output_dir)

    boxes_tgt = []
    boxes_out = []
    boxes_iou = []
    image_ids = []

    with torch.no_grad():
        for img, tgt in tqdm(loader):

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

    boxes_out = rearrange(boxes_out, "list boxes coord -> (list boxes) coord")
    boxes_tgt = rearrange(boxes_tgt, "list boxes coord -> (list boxes) coord")
    boxes_iou = rearrange(boxes_iou, "list ious -> (list ious)")
    image_ids = rearrange(image_ids, "list ids -> (list ids)")

    np.save("outputs/test_results/boxes_out.npy", boxes_out.numpy())
    np.save("outputs/test_results/boxes_tgt.npy", boxes_tgt.numpy())
    np.save("outputs/test_results/boxes_iou.npy", boxes_iou.numpy())
    np.save("outputs/test_results/image_ids.npy", image_ids.numpy())

    sys.exit(1)

if __name__ == "__main__":
    main()
