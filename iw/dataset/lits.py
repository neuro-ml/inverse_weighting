import numpy as np

from skimage import measure

from dpipe.dataset.segmentation import SegmentationFromCSV


class LITS(SegmentationFromCSV):
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
        return np.int32(np.shape(super().load_segm(identifier=identifier)))

    def load_centers(self, identifier):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        return [np.argwhere(y_bin)]

    def load_tumor_centers(self, identifier):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        labels, n_labels = measure.label(y_bin, neighbors=8, return_num=True)
        return [np.argwhere(labels == label) for label in range(1, n_labels + 1)]

    def load_cc(self, identifier, get_cc_fn):
        y_bin = np.any(self.load_segm(identifier=identifier), axis=0)
        return np.array([get_cc_fn(y_bin)], dtype='float32')


def scale_ct(x: np.ndarray, min_value: float = -300, max_value: float = 300) -> np.ndarray:
    x = np.float32(np.clip(x, a_min=min_value, a_max=max_value))
    x -= np.min(x)
    x /= np.max(x)
    return x
