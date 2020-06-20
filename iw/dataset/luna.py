import numpy as np

from skimage import measure

from dpipe.dataset import Dataset
from dpipe.dataset.segmentation import SegmentationFromCSV


class LUNA(SegmentationFromCSV):

    def __init__(self, data_path, modalities, target='target', metadata_rpath='metadata.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)

    def load_image(self, identifier):
        return np.float32(super().load_image(identifier))

    def load_segm(self, identifier):
        return np.float32(super().load_segm(identifier)[None])

    def load_shape(self, identifier):
        return np.int32(np.shape(super().load_segm(identifier=identifier))[1:])

    def load_centers(self, identifier):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        return np.argwhere(y_bin)

    def load_tumor_centers(self, identifier):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        labels, n_labels = measure.label(y_bin, neighbors=8, return_num=True)
        return [np.argwhere(labels == label) for label in range(1, n_labels + 1)]

    def load_cc(self, identifier, get_cc_fn):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        return np.array([get_cc_fn(y_bin)], dtype='float32')


def get_n_tumors(ids, df):
    return df['n_tumors'].loc[ids].values


def scale_ct(x: np.ndarray, min_value: float = -1000, max_value: float = 300) -> np.ndarray:
    x = np.clip(x, a_min=min_value, a_max=max_value)
    x -= np.min(x)
    x /= np.max(x)
    return np.float32(x)


class Proxy:
    """Base class for all wrappers."""

    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)

    def __dir__(self):
        return list(set(super().__dir__()) | set(dir(self._shadowed)))


def apply_mask(dataset: Dataset, mask_modality_id: int = -1, mask_value: int = None) -> Dataset:
    class MaskedDataset(Proxy):
        def load_image(self, patient_id):
            images = self._shadowed.load_image(patient_id)
            mask = images[mask_modality_id]

            mask_bin = mask > 0 if mask_value is None else mask == mask_value
            if not np.sum(mask_bin) > 0:
                raise ValueError('The obtained mask is empty')

            images = [image * mask for image in images[:-1]]
            for image in images:
                fill_value = np.min(image)
                image[mask == 0] = fill_value

            return np.array(images)

        @property
        def n_chans_image(self):
            return self._shadowed.n_chans_image - 1

    return MaskedDataset(dataset)
