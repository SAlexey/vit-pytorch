#%%
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from einops import parse_shape, rearrange, repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.resnets import resnet18_3d, resnet50_3d
from data import transforms as T
from data.oai import MOAKSDataset
from util.box_ops import box_cxcywh_to_xyxy, box_iou


#%%


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="/scratch/htc/ashestak/oai/v00/data"
    )
    parser.add_argument(
        "--anns", type=str, default="/scratch/htc/ashestak/oai/v00/data/moaks/test.json"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=("resnet18_3d", "resnet50_3d")
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--weights", type=str, required=True)
    return parser.parse_args()


MODELS = {
    "resnet18_3d": resnet18_3d,
    "resnet50_3d": resnet50_3d,
}


def main(args):

    assert args.model in MODELS, f"Unknown model '{args.model}'"

    model = MODELS.get(args.model)

    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    model.eval()

    anns = Path(args.anns)
    root = Path(args.data_root)

    assert anns.exists(), "Provided path for annotations does not exist!"
    assert root.exists(), "Provided path for data does not exist!"

    transforms = T.Compose((T.ToTensor(), T.Normalize(target=False)))

    data = MOAKSDataset(root, anns, transforms=transforms)

    loader = DataLoader(data, batch_size=1, num_workers=2)

    output_dir = args.output_dir

    if not output_dir:
        output_dir = f"./outputs/{args.model}/test"

    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = []
    scale_fct = []
    boxes_out = []
    boxes_tgt = []
    class_out = []
    class_tgt = []

    with torch.no_grad():
        for img, tgt in tqdm(loader):

            out = model(img)

            out_box = out["boxes"]
            out_cls = out["labels"]

            tgt_box = tgt["boxes"]
            tgt_cls = tgt["labels"]
            tgt_ids = tgt["image_id"]

            fct = torch.as_tensor(img.shape[-3:])

            fct = repeat(fct, "shape -> bs (repeat shape)", bs=1, repeat=2)

            obj_shape = parse_shape(tgt_box, "batch objects coord")

            out_box = rearrange(
                out_box, "batch (objects coord) -> (batch objects) coord", **obj_shape
            )

            tgt_box = rearrange(
                tgt_box, "batch objects coord -> (batch objects) coord", **obj_shape
            )

            obj_shape.pop("coord")

            tgt_ids = repeat(tgt_ids, "(batch ids) -> (batch objects ids)", **obj_shape)

            image_ids.append(tgt_ids)
            class_out.append(out_cls)
            class_tgt.append(tgt_cls)
            boxes_out.append(out_box)
            boxes_tgt.append(tgt_box)

    boxes_out = rearrange(boxes_out, "list objects coord -> (list objects) coord")
    boxes_tgt = rearrange(boxes_tgt, "list objects coord -> (list objects) coord")
    class_out = rearrange(class_out, "list objects class -> (list objects) class")
    class_tgt = rearrange(class_tgt, "list objects class -> (list objects) class")
    image_ids = rearrange(image_ids, "list ids -> (list ids)")

    np.save("outputs/test_results/boxes_out.npy", boxes_out.numpy())
    np.save("outputs/test_results/boxes_tgt.npy", boxes_tgt.numpy())
    np.save("outputs/test_results/image_ids.npy", image_ids.numpy())

    sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
