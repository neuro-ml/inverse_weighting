from functools import partial

import torch
import torch.nn as nn

from dpipe.layers.fpn import FPN
from dpipe.layers.resblock import ResBlock3d
from dpipe.layers.conv import PreActivation3d


def get_unet(n_chans_in, n_chans_out):
    res_block = partial(ResBlock3d, kernel_size=3, padding=1)
    shortcut = partial(PreActivation3d, kernel_size=1)
    upsample = partial(nn.Upsample, scale_factor=2, mode='trilinear', align_corners=True)
    downsample = partial(nn.MaxPool3d, kernel_size=2)

    structure = [
        [[16, 16, 16], shortcut(16, 16), [16, 16, 16]],
        [[16, 64, 64], shortcut(64, 64), [64, 64, 16]],
        [[64, 128, 128], shortcut(128, 128), [128, 128, 64]],
        [[128, 256, 256, 128]]
    ]

    init_path = nn.Sequential(
        nn.Conv3d(n_chans_in, 16, kernel_size=3, padding=1, bias=False),
        PreActivation3d(16, 16, kernel_size=3, padding=1)
    )

    res16 = FPN(
        res_block, downsample, upsample, torch.add,
        structure, kernel_size=3, dilation=1, padding=1, last_level=True
    )

    out_path = nn.Sequential(
        ResBlock3d(16, 16, kernel_size=1),
        PreActivation3d(16, n_chans_out, kernel_size=1, bias=False),
        nn.BatchNorm3d(n_chans_out)
    )

    architecture = nn.Sequential(init_path, res16, out_path)
    return architecture
