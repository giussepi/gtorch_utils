# -*- coding: utf-8 -*-
"""
gtorch_utils/nns/models/segmentation/deeplab/operators/aspp.py

Source https://github.com/YudeWang/semantic-segmentation-codebase/blob/main/lib/net/operators/ASPP.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from gtorch_utils.nns.utils.sync_batchnorm import SynchronizedBatchNorm2d


__all__ = ['ASPP']


class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling (ASPP) Module  """

    def __init__(
            self, dim_in, dim_out, rate=[1, 6, 12, 18], bn_mom=0.1, has_global=True,
            batchnorm=SynchronizedBatchNorm2d
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_global = has_global
        if rate[0] == 0:
            self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=1, bias=False),
                batchnorm(dim_out, momentum=bn_mom, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[0], dilation=rate[0], bias=False),
                batchnorm(dim_out, momentum=bn_mom, affine=True),
                nn.ReLU(inplace=True),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[1], dilation=rate[1], bias=False),
            batchnorm(dim_out, momentum=bn_mom, affine=True),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[2], dilation=rate[2], bias=False),
            batchnorm(dim_out, momentum=bn_mom, affine=True),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate[3], dilation=rate[3], bias=False),
            batchnorm(dim_out, momentum=bn_mom, affine=True),
            nn.ReLU(inplace=True),
        )
        if self.has_global:
            self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            self.branch5_bn = batchnorm(dim_out, momentum=bn_mom, affine=True)
            self.branch5_relu = nn.ReLU(inplace=True)
            self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=False),
                batchnorm(dim_out, momentum=bn_mom, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        else:
            self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
                batchnorm(dim_out, momentum=bn_mom, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )

    def forward(self, x):
        result = None
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        if self.has_global:
            global_feature = F.adaptive_avg_pool2d(x, (1, 1))
            global_feature = self.branch5_conv(global_feature)
            global_feature = self.branch5_bn(global_feature)
            global_feature = self.branch5_relu(global_feature)
            global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', align_corners=False)

            feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        else:
            feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)

        return result
