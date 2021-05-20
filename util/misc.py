from numbers import Number
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch


def save_as_nifty(arr, name):
    img = nib.Nifti1Image(arr.astype("uint8"), np.eye(4))
    img.to_filename(name)


def _is_sequence(obj):
    return hasattr(obj, "__len__") and hasattr(obj, "__iter__")


def _is_numeric(obj):
    if isinstance(obj, Number):
        return True
    elif _is_sequence(obj) and not isinstance(obj, (str, dict)):
        return all([_is_numeric(each) for each in obj])
    else:
        return False


class DCMSeries(object):
    def __init__(self, path=None):

        self.path = path
        self.image = None

        self.reader = sitk.ImageSeriesReader()

        if self.path is not None:
            self._read(path)

    @property
    def array(self):
        return sitk.GetArrayFromImage(self.image)

    @property
    def tensor(self):
        return torch.as_tensor(self.array)

    def numpy(self):
        return self.array

    def _read(self, path):
        file_names = self.reader.GetGDCMSeriesFileNames(path)
        self.reader.SetFileNames(file_names)
        self.image = self.reader.Execute()
        return self.image

    def __call__(self, path, new=False):
        if new:
            return DCMSeries(path)
        self._read(path)
        return self
