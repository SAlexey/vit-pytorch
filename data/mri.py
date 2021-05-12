import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

import SimpleITK as sitk
from scipy import ndimage
from sklearn.model_selection import train_test_split


class MRIDatasetBase(object):
    """Base Mixin for all mri datasets"""

    def get_coco_api(self, *args, **kwargs):
        raise NotImplementedError()


class DICOMDataset(Dataset):
    reader = sitk.ImageSeriesReader()

    def __init__(self, root, anns, transforms=None):
        self.root = Path(root)
        with open(anns) as fh:
            self.anns = json.load(fh)

        self.keys = sorted(list(self.anns.keys()))
        self.transform = transforms

    def __len__(self):
        return len(self.keys)


class DICOMDatasetMasks(DICOMDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # only take annotations with 2 bounding boxes
        self.keys = [k for k in self.keys if len(self.anns[k]["boxes"]) == 2]

    def __getitem__(self, idx):
        key = self.keys[idx]
        ann = self.anns[key]

        mask_path = (
            Path(
                ann["dicom_dir"].replace(
                    "oaiDataBase", "bzftacka/OAI_DESS_Data_AllTPs/Merged"
                )
            )
            / "Segmentation"
        )

        file_names = self.reader.GetGDCMSeriesFileNames(str(mask_path))
        self.reader.SetFileNames(file_names)
        mask = self.reader.Execute()
        mask = sitk.GetArrayFromImage(mask)

        boxes = []

        objects = ndimage.find_objects(mask)
        for label, obj in enumerate(objects, 1):
            if obj is not None and label in (5, 6):
                xs, ys, zs = obj
                box = [
                    (xs.start + xs.stop) / 2.0,
                    (ys.start + ys.stop) / 2.0,
                    (zs.start + zs.stop) / 2.0,
                    xs.stop - xs.start,
                    ys.stop - ys.start,
                    zs.stop - zs.start,
                ]
                boxes.append(box)

        mask = torch.from_numpy(mask)
        boxes = torch.as_tensor(boxes) / torch.as_tensor(mask.shape).repeat(1, 2)

        tgt = {"image_id": torch.as_tensor(ann.get("image_id")), "boxes": boxes}

        return mask.unsqueeze(0), tgt


class MOAKSDataset(MRIDatasetBase, Dataset):
    def __init__(self, root, anns, transforms=None):
        self.root = root
        self.transform = transforms


class MRIDataset(MRIDatasetBase, Dataset):
    def __init__(self, root, anns, transforms=None):
        """
        Map style dataset for 3D MRI
        Arguments:
            img_root: (str|Path) image directory
            ann_file: (str|Path) target annotation  file location
            transforms: (callable) item transforms
        """
        with open(anns) as fh:
            self.anns = json.load(fh)

        self.keys = list(self.anns.keys())

        self.root = Path(root)
        self.transform = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        prepare and return image + target

        returns:
            img: (tensor)
            tgt: (dict) with keys:
                - boxes: (tensor) box in xyxy format
                - labels: (tensor) class labels
                - image_id: (int)
                - side: (str) leg side
        """
        key = self.keys[idx]
        ann = self.anns[key]

        tgt = {
            "image_id": ann.get("image_id", 0),
            "boxes": ann.get("boxes", []),
            "labels": ann.get("labels", []),
            "iscrowd": [
                0,
            ]
            * len(ann.get("boxes", [])),
            "side": int(ann.get("side") == "right"),
        }

        # load the image
        img = np.load(self.root / ann.get("file_name", f"{tgt['image_id']}.npy"))

        if self.transform is not None:
            img, tgt = self.transform(img, tgt)

        return img, tgt

    def get_coco_api(self):

        imgs = []
        anns = []
        info = {}
        cats = {}
        cids = set()

        ann_id = 0

        # for k, ann in self.anns.items():


def make_mri_transforms(image_set):

    scales = [124, 164, 200, 228, 344, 480]

    if image_set == "train":
        return T.Compose(
            [
                T.ToTensor(),
                # T.RandomHorizontalFlip(),
                # T.RandomResize(scales, max_size=512),
                # T.RandomSelect(
                # T.Compose(
                #    [
                #        T.RandomResize([400, 500]),
                #        T.RandomSizeCrop(384, 480),
                #        T.RandomResize(scales, max_size=512),
                #    ]
                # ),
                # ),
                T.Normalize(mean=[0.459], std=[0.229]),
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.ToTensor(),
                # T.RandomResize([384], max_size=400),
                T.Normalize(mean=[0.459], std=[0.229]),
            ]
        )


def build_moaks(image_set, args):
    pass


def build_mri(image_set, args):

    root = Path(args.root)
    assert root.exists(), f"provided dataset path {root} does not exist"

    PATHS = {
        "train": (root / "inputs" / "train", root / "annotations" / "train.json"),
        "val": (root / "inputs" / "train", root / "annotations" / "val.json"),
        "test": (root / "inputs" / "test", root / "annotations" / "test.json"),
    }

    img_folder, ann_file = PATHS[image_set]

    return MRIDataset(img_folder, ann_file, transforms=make_mri_transforms(image_set))


if __name__ == "__main__":

    root = "/scratch/visual/ashestak/oai/v00/data/inputs/train"
    anns = "/scratch/visual/ashestak/oai/v00/data/moaks/val.json"
    dataset = MRIDataset(root, anns)
    img, tgt = next(iter(dataset))
    print(img.shape)
    print(tgt)
