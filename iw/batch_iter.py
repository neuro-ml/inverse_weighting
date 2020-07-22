import random

import numpy as np

from dpipe.medim.patch import sample_box_center_uniformly
from dpipe.medim.box import get_centered_box
from dpipe.medim.shape_ops import crop_to_box


SPATIAL_DIMS = (-3, -2, -1)


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def center_choice(inputs, y_patch_size, nonzero_fraction=0.5, tumor_sampling=True, with_cc=False):
    if with_cc:
        x, y, cc, centers = inputs
    else:
        x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    if len(centers) > 0 and np.random.uniform() < nonzero_fraction:
        center = random.choice(random.choice(centers)) if tumor_sampling else random.choice(centers)
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)

    if with_cc:
        return x, y, cc, center
    else:
        return x, y, center


def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
    weight = np.zeros_like(cc, dtype='float32')
    cc_items = np.unique(cc)
    K = len(cc_items) - 1
    N = np.prod(cc.shape)
    for i in cc_items:
        weight[cc == i] = N / ((K + 1) * np.sum(cc == i))
    # we add clip for learning stability
    # e.g. if we catch only 1 voxel of some component, the corresponding weight will be extremely high (~1e6)
    return np.clip(weight, w_min, w_max)


def extract_patch(inputs, x_patch_size, y_patch_size, with_cc=False):
    if with_cc:
        x, y, cc, center = inputs
    else:
        x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0)

    if with_cc:
        cc_patch = crop_to_box(cc, box=y_spatial_box, padding_values=0)
        w_patch = cc2weight(cc_patch)
        return x_patch, y_patch, w_patch
    else:
        return x_patch, y_patch
