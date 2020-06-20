import os
from os.path import join as jp
from lxml import etree

import numpy as np


def id2root(xml_data_path, _id):
    """Returns an ``etree`` root from xml entry corresponding to given ``_id``."""
    for int_dir in os.listdir(xml_data_path):
        int_path = jp(xml_data_path, int_dir)
        for xml_rpath in os.listdir(int_path):
            xml_path = jp(int_path, xml_rpath)

            # read xml tree:
            root = etree.parse(xml_path).getroot()

            # searching for ResponseHeader:
            for child in root:
                if not isinstance(child, etree._Comment):
                    if 'ResponseHeader' in child.tag:
                        header_child = child
                        break

            # comparing `_id` with current xml's id:
            for child in header_child:
                if '{http://www.nih.gov}SeriesInstanceUid' == child.tag and _id == child.text:
                    return root


def root2expert_roots(root):
    """Returns list of roots corresponding to all experts sessions."""
    rs_children = []

    for child in root:
        if not isinstance(child, etree._Comment):
            if 'readingSession' in child.tag:
                rs_children.append(child)

    return rs_children


def expert_root2nodules(expert_root):
    """
    Returns ``nodules`` array from given expert root. Array has following structure:

    nodules = [nodule_0, nodule_1, ...]
    nodule_N = [roi_0, roi_1, ...]
    roi_M = [z_coord, [xy_indexes_0, xy_indexes_1, ...]]
    xy_indexes_K = [x_index, y_index]
    """
    nodules = []
    for child1 in expert_root:

        rois = []
        if 'unblindedReadNodule' in child1.tag:  # in nodule

            for child2 in child1:

                if not isinstance(child2, etree._Comment):
                    if 'roi' in child2.tag:

                        z_coord, xy_indexes = 0, []
                        for child3 in child2:  # in nodule roi
                            if 'imageZposition' in child3.tag:
                                z_coord = float(child3.text)
                            if 'edgeMap' in child3.tag:

                                xy_index = [0, 0]
                                for child4 in child3:  # in edge map
                                    if 'xCoord' in child4.tag:
                                        xy_index[0] = int(child4.text)
                                    if 'yCoord' in child4.tag:
                                        xy_index[1] = int(child4.text)
                                xy_indexes.append(xy_index)

                        roi = [z_coord, xy_indexes]
                        rois.append(roi)

        if len(rois) > 0:
            nodules.append(rois)

    return nodules


def nodules2centers(nodules, z_origin, z_spacing):
    centers = []
    for nodule in nodules:
        nodule_indexes = []
        for roi in nodule:
            z = (roi[0] - z_origin) / z_spacing
            for xy in roi[1]:
                nodule_indexes.append([xy[0], xy[1], z])
        centers.append(np.mean(nodule_indexes, axis=0))

    return np.round(centers)


def get_nodules(_id, xml_dp):
    """Gets given expert delineation."""
    root = id2root(xml_dp, _id)
    expert_roots = root2expert_roots(root)
    expert_nodules = [expert_root2nodules(expert_root) for expert_root in expert_roots]
    return expert_nodules
