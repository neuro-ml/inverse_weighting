import numpy as np
import cv2

from skimage import measure


def fill3d(img3d, nodule, z_origin, z_spacing):
    """Fills LUNA target ``img3d`` with given ``nodule`` roi."""
    img3d = np.float32(img3d)

    for roi in nodule:
        z = int((roi[0] - z_origin) / z_spacing)
        pts = np.int32([roi[1]])
        img = np.zeros_like(img3d[..., z], dtype='float32').T

        img3d[::, ::, z] += cv2.fillPoly(img.copy(), pts, 1).T

    return np.clip(img3d, 0, 1)


def get_connected_components(y):
    cc = measure.label(y, neighbors=8)
    return np.array(cc, dtype='float32')


def scale_flair(images, apply_at=0):
    images = images.astype(np.float32)

    img_to_scale = images[apply_at]

    lower_th = np.percentile(img_to_scale, 50)
    upper_th = np.percentile(img_to_scale, 99)

    img_clipped = np.clip(img_to_scale, a_min=lower_th, a_max=upper_th)
    img_zeroed = img_clipped - np.min(img_clipped)
    img_scaled = img_zeroed / np.max(img_zeroed)

    images[apply_at] = img_scaled
    return images


def interpolate_np(x, scale_factor, axes=(-1, -2, -3)):
    for ax in axes:
        x = np.repeat(x, scale_factor, axis=ax)
    return x
