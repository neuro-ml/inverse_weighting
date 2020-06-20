import os
from os.path import join as jp

import numpy as np
import nibabel as nb
import SimpleITK as sitk

from dpipe.io import load_pred


def load_pred_from_exp(_id, exp_path, n_exp=5):
    """Finds and loads the ``_id`` prediction in ``n_exp`` validation series of ``exp_path`` experiment."""
    pred = None

    for n in range(n_exp):
        try:
            pred = load_pred(_id, jp(exp_path, f'experiment_{n}/test_predictions'))
        except FileNotFoundError:
            pass

    if pred is None:
        raise FileNotFoundError(f'There is no such id `{_id}` over {n_exp} experiments.')

    return pred


def load_nii(path_to_file):
    """Loads image stored in nifty format from ``path_to_file``."""
    img = nb.load(path_to_file).get_data()
    return img


def save_nii(path_to_file, img):
    """Saves ``img`` of type `numpy.ndarray` in nifty format."""
    nb.save(nb.Nifti1Image(img, np.eye(4)), path_to_file)


def load_itkimage_ct(_id, data_path):
    """Loader for LUNA16_raw CT images."""
    itkimage = None

    n_subsets = 10
    for i in range(n_subsets):
        try:
            filename = jp(jp(data_path, f'subset{i}'), _id + '.mhd')
            itkimage = sitk.ReadImage(filename)
        except RuntimeError:
            pass

    if itkimage is None:
        raise FileNotFoundError(f'There is no such id `{_id}`.')

    return itkimage


def load_itkimage_lungmask(_id, lungmask_path):
    filename = jp(lungmask_path, _id + '.mhd')
    itkimage = sitk.ReadImage(filename)
    return itkimage


def get_iw_dir_name():
    return os.path.dirname(__file__)


def get_ids(data_path, subset_ids):
    """Gets ids from the given files' names"""
    ids = []
    for subset_id in subset_ids:
        ids_dp = os.listdir(jp(data_path, f'subset{subset_id}'))
        [(ids.append(id_dp.strip('.mhd')) if id_dp.endswith('mhd') else None) for id_dp in ids_dp]

    return ids


def itkimage2image(itkimage):
    """Moves z-axis to the last position."""
    return np.swapaxes(sitk.GetArrayFromImage(itkimage), 0, 2)
