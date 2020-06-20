import os
import random

import numpy as np
import SimpleITK as sitk
import torch


def bin_segm(segm, bin_th=None):
    if bin_th is not None:
        return np.array(segm > bin_th, dtype=segm.dtype)
    else:
        return segm


def get_pred(x, threshold=0.5):
    return x[0] > threshold


def np_sigmoid(x):
    """Applies sigmoid function to the incoming value(-s)."""
    return 1 / (1 + np.exp(-x))


def get_positive_class_fraction(ids, load_y):
    lesion_v = 0
    total_v = 0

    for _id in ids:
        segm = get_pred(load_y(_id))
        lesion_v += np.sum(segm)
        total_v += np.prod(segm.shape)

    return lesion_v / total_v


def volume2diameter(volume):
    return (6 * volume / np.pi) ** (1 / 3)


def itkimage2image(itkimage):
    return np.swapaxes(sitk.GetArrayFromImage(itkimage), 0, 2)


def is_right_shape(scan):
    shape = scan.shape
    new_shape = np.array(shape)

    for dim in (-3, -2, -1):
        gap = shape[dim] % 8
        if gap != 0:
            new_shape[dim] -= gap

    return scan[tuple(map(slice, new_shape))]


def get_iw_dir_name():
    return os.path.dirname(__file__)


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
